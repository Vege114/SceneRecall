import type { CharacterConfirmationRole } from '../identityTypes';

const detectionMessages = {
  missing: '尚无本地匹配参考。先缓存来源参考图，或在角色参考图库添加清晰的单人图片。',
  usable: '已有检测可用的单人参考图，匹配候选仍需人工核对。',
  not_checked: '参考图尚未检测，暂不能判断能否用于匹配。可保存名单并生成候选以进行本地检测。',
  failed: '当前参考图未得到可用的单人人物检测，暂未参与角色匹配。可在角色参考图库更换清晰单人图，或从待判定池选择角色后人工核对。',
  model_unavailable: '人物检测模型暂不可用，无法判断参考图能否用于匹配。请检查设置中的本地人物检测模型后重试生成。',
};
const issueLabels = { not_checked: '尚未检测', no_person: '未检测到人物', multiple_people: '检测到多个人物', model_unavailable: '检测模型不可用' };

export default function ReferenceDiagnostics({ role }: { role: CharacterConfirmationRole }) {
  if (!role.reference_detection_status) return null;
  return <section className="helper" aria-label={`${role.profile.name}的参考图检测`}>
    <p role="status">本地匹配参考 {role.matching_reference_count ?? role.trusted_reference_count} 张 · 检测可用 {role.usable_matching_reference_count ?? 0} 张。{detectionMessages[role.reference_detection_status]}</p>
    {role.reference_model && <p>检测模型：{role.reference_model}</p>}
    {!!role.reference_issues?.length && <ul>{role.reference_issues.map((issue) => <li key={issue.reference_id}>
      <strong>{issueLabels[issue.status]}</strong>{issue.reason && <span>：{issue.reason}</span>}{issue.action && <p>下一步：{issue.action}</p>}
    </li>)}</ul>}
    {role.reference_detection_status === 'failed' && <p>参考图已缓存不等于人物检测通过；增加正片采样不会修复参考图。没有可用参考或匹配候选不代表该角色未出场。</p>}
    {role.reference_detection_status === 'not_checked' && <p>尚未检测不表示参考图无效，也不表示该角色未出场。</p>}
    {role.reference_detection_status === 'model_unavailable' && <p>模型不可用不表示参考图错误，无需因此撤销参考图或已保存的人工标注。</p>}
  </section>;
}
