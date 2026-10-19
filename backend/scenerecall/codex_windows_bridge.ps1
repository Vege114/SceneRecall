$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Console]::InputEncoding = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

# The fixed launcher receives data on stdin. No request text is evaluated as
# PowerShell, and ProcessStartInfo bypasses cmd.exe and shell associations.
$request = [Console]::In.ReadToEnd() | ConvertFrom-Json
Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;

public sealed class SceneRecallJob : IDisposable {
    [StructLayout(LayoutKind.Sequential)] struct BasicLimit {
        public long ProcessTime, JobTime;
        public uint Flags;
        public UIntPtr MinimumWorkingSet, MaximumWorkingSet;
        public uint ActiveProcesses;
        public UIntPtr Affinity;
        public uint PriorityClass, SchedulingClass;
    }
    [StructLayout(LayoutKind.Sequential)] struct IoCounters {
        public ulong ReadOps, WriteOps, OtherOps, ReadBytes, WriteBytes, OtherBytes;
    }
    [StructLayout(LayoutKind.Sequential)] struct ExtendedLimit {
        public BasicLimit Basic;
        public IoCounters Io;
        public UIntPtr ProcessMemory, JobMemory, PeakProcessMemory, PeakJobMemory;
    }
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    static extern IntPtr CreateJobObject(IntPtr attributes, string name);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool SetInformationJobObject(IntPtr job, int type, IntPtr info, uint length);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr handle);
    IntPtr handle;
    public SceneRecallJob() {
        handle = CreateJobObject(IntPtr.Zero, null);
        if (handle == IntPtr.Zero) throw new Win32Exception();
        ExtendedLimit limits = new ExtendedLimit();
        limits.Basic.Flags = 0x2000; // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        int size = Marshal.SizeOf(typeof(ExtendedLimit));
        IntPtr memory = Marshal.AllocHGlobal(size);
        try {
            Marshal.StructureToPtr(limits, memory, false);
            if (!SetInformationJobObject(handle, 9, memory, (uint)size)) throw new Win32Exception();
        } catch { Dispose(); throw; }
        finally { Marshal.FreeHGlobal(memory); }
    }
    public void Add(Process process) {
        if (!AssignProcessToJobObject(handle, process.Handle)) throw new Win32Exception();
    }
    public void Dispose() {
        if (handle != IntPtr.Zero) { CloseHandle(handle); handle = IntPtr.Zero; }
    }
}
'@

$allowed = @('PATH', 'HOME', 'CODEX_HOME', 'USER', 'LOGNAME', 'TMPDIR', 'TEMP', 'TMP',
    'SYSTEMROOT', 'WINDIR', 'PATHEXT', 'COMSPEC', 'APPDATA', 'LOCALAPPDATA',
    'USERPROFILE', 'HOMEDRIVE', 'HOMEPATH', 'USERNAME',
    'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY', 'SSL_CERT_FILE', 'SSL_CERT_DIR',
    'LANG', 'LC_ALL', 'LC_CTYPE')
$start = New-Object System.Diagnostics.ProcessStartInfo
$start.FileName = [string]$request.executable
$start.Arguments = [string]$request.arguments
$start.WorkingDirectory = [string]$request.directory
$start.UseShellExecute = $false
$start.CreateNoWindow = $true
$start.RedirectStandardInput = $true
$start.RedirectStandardOutput = $true
$start.RedirectStandardError = $true
$start.EnvironmentVariables.Clear()
foreach ($entry in [Environment]::GetEnvironmentVariables().GetEnumerator()) {
    if ($allowed -contains $entry.Key.ToUpperInvariant()) {
        $start.EnvironmentVariables[$entry.Key] = [string]$entry.Value
    }
}
foreach ($entry in $request.environment.PSObject.Properties) {
    if ($allowed -contains $entry.Name.ToUpperInvariant()) {
        $start.EnvironmentVariables[$entry.Name] = [string]$entry.Value
    }
}

$job = New-Object SceneRecallJob
$child = New-Object System.Diagnostics.Process
$child.StartInfo = $start
$exitCode = 1
try {
    if (-not $child.Start()) { throw 'Unable to start Codex CLI' }
    try { $job.Add($child) } catch { $child.Kill(); throw }
    $stdout = $child.StandardOutput.BaseStream.CopyToAsync([Console]::OpenStandardOutput())
    $stderr = $child.StandardError.BaseStream.CopyToAsync([Console]::OpenStandardError())
    $inputBytes = [Convert]::FromBase64String([string]$request.input)
    try { $child.StandardInput.BaseStream.Write($inputBytes, 0, $inputBytes.Length) }
    catch [System.IO.IOException] { } # Child may reject a request before reading stdin.
    finally { $child.StandardInput.Close() }
    $child.WaitForExit()
    $exitCode = $child.ExitCode
    [System.Threading.Tasks.Task]::WaitAll(@($stdout, $stderr))
} finally {
    $job.Dispose()
    $child.Dispose()
}
exit $exitCode
