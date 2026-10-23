import { useEffect, useRef, type ReactNode } from 'react';
import { AlertCircle, Check, Film, LoaderCircle, X } from 'lucide-react';

export function Spinner({ label = '正在加载' }: { label?: string }) {
  return <span className="loading" role="status"><LoaderCircle size={18} className="spin" /><span>{label}</span></span>;
}
export function Alert({ children, tone = 'warning' }: { children: ReactNode; tone?: 'warning' | 'error' | 'info' | 'success' }) {
  return <div className={`alert alert-${tone}`} role={tone === 'error' ? 'alert' : 'status'}>{tone === 'success' ? <Check size={17} /> : <AlertCircle size={17} />}<div>{children}</div></div>;
}
export function Empty({ icon, title, children, action }: { icon?: ReactNode; title: string; children?: ReactNode; action?: ReactNode }) {
  return <div className="empty"><div className="empty-icon">{icon || <Film size={28} strokeWidth={1.4} />}</div><h3>{title}</h3>{children && <p>{children}</p>}{action}</div>;
}
export function PageHeader({ eyebrow, title, children, action }: { eyebrow: string; title: string; children?: ReactNode; action?: ReactNode }) {
  return <header className="page-heading"><div><span className="eyebrow">{eyebrow}</span><h1>{title}</h1>{children && <p>{children}</p>}</div>{action && <div className="heading-actions">{action}</div>}</header>;
}
export function Modal({ title, subtitle, children, onClose, wide = false }: { title: string; subtitle?: string; children: ReactNode; onClose: () => void; wide?: boolean }) {
  const dialog = useRef<HTMLDivElement>(null);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;
  useEffect(() => {
    const previous = document.activeElement as HTMLElement;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    const focusables = () => Array.from(dialog.current?.querySelectorAll<HTMLElement>('button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), a[href]') || []);
    const first = focusables()[0]; first?.focus();
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') closeRef.current();
      if (e.key === 'Tab') {
        const elements = focusables();
        const first = elements[0]; const last = elements[elements.length - 1];
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last?.focus(); }
        if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first?.focus(); }
      }
    };
    document.addEventListener('keydown', handleKey);
    return () => { document.body.style.overflow = previousOverflow; document.removeEventListener('keydown', handleKey); previous?.focus(); };
  }, []);
  return <div className="modal-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}><div ref={dialog} role="dialog" aria-modal="true" aria-label={title} className={`modal ${wide ? 'modal-wide' : ''}`}><div className="modal-heading"><div><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div><button type="button" className="icon-button" aria-label="关闭" onClick={onClose}><X size={21} /></button></div>{children}</div></div>;
}
export function Field({ label, hint, children, className = '' }: { label: string; hint?: string; children: ReactNode; className?: string }) {
  return <label className={`field ${className}`}><span className="field-label">{label}</span>{children}{hint && <span className="field-hint">{hint}</span>}</label>;
}
export function Status({ status }: { status?: string }) {
  const text: Record<string, string> = { core_ready: '核心检索就绪', existing: '已在资料库', registered: '待分析', imported: '已导入', pending: '等待中', queued: '排队中', waiting_input: '等待角色确认', running: '处理中', paused: '已暂停', pausing: '暂停中', completed: '已完成', complete: '已完成', cancelling: '取消中', cancelled: '已取消', canceled: '已取消', failed: '失败', ready: '已就绪', analyzed: '已分析', partial: '部分完成', interrupted: '已中断', unreviewed: '待复核' };
  return <span className={`status status-${status || 'registered'}`}><span />{text[status || 'registered'] || status}</span>;
}
export function Thumbnail({ src, title = '', className = '' }: { src?: string; title?: string; className?: string }) {
  return <div className={`thumbnail ${className}`}>{src ? <img src={src} alt={title} loading="lazy" onError={(event) => { event.currentTarget.style.display = 'none'; }} /> : <Film size={28} strokeWidth={1} />}<div className="thumbnail-grain" /></div>;
}
