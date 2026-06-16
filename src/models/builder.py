from src.models.hatr import BaseClassifier
from src.models.hatr_proxy_anchor import BaseClassifierProxyAnchor
from src.models.hatr_proxy_anchor_deep_shared import BaseClassifierProxyAnchorDeepShared
from src.models.hatr_proxy_anchor_simple import BaseClassifierProxyAnchorSimple

def _normalize_run_mode(run_mode: str | None) -> str:
    if run_mode is None:
        return "standard"
    normalized = str(run_mode).strip().lower()
    aliases = {
        "base": "standard",
        "classifier": "standard",
        "proxy": "proxy_only",
        "proxy_anchor": "proxy_only",
        "proxy-anchor": "proxy_only",
        "dual": "dual_head",
        "dualhead": "dual_head",
    }
    return aliases.get(normalized, normalized)


def build_model(config: dict, run_mode: str | None = None, **kwargs):
    model_cfg = config.get('model', {}) if isinstance(config.get('model'), dict) else {}
    resolved_run_mode = _normalize_run_mode(run_mode or model_cfg.get('run_mode'))
    model_name = str(model_cfg.get('name', '')).strip().lower()
    use_simple_proxy = any(
        key in model_name
        for key in ('proxy_anchor_simple', 'proxy_anchor_simplified', 'proxy_anchor_lite')
    )
    use_deep_shared_proxy = any(
        key in model_name
        for key in ('proxy_anchor_deep_shared', 'proxy_anchor_shared', 'proxy_anchor_no_projector')
    )

    if resolved_run_mode == "standard":
        kwargs = dict(kwargs)
        kwargs.pop('use_classifier', None)
        print("Model Builder: Instantiating BaseClassifier (run_mode=standard)")
        return BaseClassifier(**kwargs)

    if resolved_run_mode == "proxy_only":
        kwargs = dict(kwargs)
        kwargs['use_classifier'] = False
        if 'embedding_dim' not in kwargs and 'embedding_dim' in model_cfg:
            kwargs['embedding_dim'] = model_cfg['embedding_dim']
        if 'normalize_embedding' not in kwargs and 'normalize_embedding' in model_cfg:
            kwargs['normalize_embedding'] = model_cfg['normalize_embedding']
        if use_deep_shared_proxy:
            print("Model Builder: Instantiating BaseClassifierProxyAnchorDeepShared (run_mode=proxy_only)")
            return BaseClassifierProxyAnchorDeepShared(**kwargs)
        if use_simple_proxy:
            print("Model Builder: Instantiating BaseClassifierProxyAnchorSimple (run_mode=proxy_only)")
            return BaseClassifierProxyAnchorSimple(**kwargs)
        print("Model Builder: Instantiating BaseClassifierProxyAnchor (run_mode=proxy_only)")
        return BaseClassifierProxyAnchor(**kwargs)

    if resolved_run_mode == "dual_head":
        kwargs = dict(kwargs)
        kwargs['use_classifier'] = True
        if 'embedding_dim' not in kwargs and 'embedding_dim' in model_cfg:
            kwargs['embedding_dim'] = model_cfg['embedding_dim']
        if 'normalize_embedding' not in kwargs and 'normalize_embedding' in model_cfg:
            kwargs['normalize_embedding'] = model_cfg['normalize_embedding']
        if use_deep_shared_proxy:
            print("Model Builder: Instantiating BaseClassifierProxyAnchorDeepShared (run_mode=dual_head)")
            return BaseClassifierProxyAnchorDeepShared(**kwargs)
        if use_simple_proxy:
            print("Model Builder: Instantiating BaseClassifierProxyAnchorSimple (run_mode=dual_head)")
            return BaseClassifierProxyAnchorSimple(**kwargs)
        print("Model Builder: Instantiating BaseClassifierProxyAnchor (run_mode=dual_head)")
        return BaseClassifierProxyAnchor(**kwargs)

    raise ValueError(
        f"Unsupported run_mode='{resolved_run_mode}'. Expected one of: standard, proxy_only, dual_head."
    )
