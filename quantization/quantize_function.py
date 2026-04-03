import torch

class AimetTraceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.runtime_batch = None

    def set_batch(self, batch):
        self.runtime_batch = batch

    def forward(self, _dummy):
        batch = self.runtime_batch
        assert batch is not None, "Batch not set"

        img = batch["img"]
        img_metas = batch["img_metas"]

        if isinstance(img, list):
            assert len(img) == 1, f"Unexpected img list length: {len(img)}"
            img = img[0]

        feats = self.model.extract_feat(img=img, img_metas=img_metas)

        # AIMET needs a tensor → pick one
        if isinstance(feats, (list, tuple)):
            for f in feats:
                if torch.is_tensor(f):
                    return f
        elif torch.is_tensor(feats):
            return feats

        raise RuntimeError("extract_feat did not return tensor")
    
    @staticmethod
    def _make_traceable_output(out):
        if torch.is_tensor(out):
            return out

        if isinstance(out, dict):
            tensor_dict = {k: v for k, v in out.items() if torch.is_tensor(v)}
            if len(tensor_dict) == 1:
                return next(iter(tensor_dict.values()))
            if tensor_dict:
                return tensor_dict

        if isinstance(out, (list, tuple)):
            gathered = []
            for item in out:
                if torch.is_tensor(item):
                    gathered.append(item)
                elif isinstance(item, dict):
                    for value in item.values():
                        if torch.is_tensor(value):
                            gathered.append(value)
            if len(gathered) == 1:
                return gathered[0]
            if gathered:
                return tuple(gathered)

        raise RuntimeError("Model output does not expose tensors suitable for AIMET tracing.")

def aimet_forward_fn(model, inputs):
    return model(torch.zeros(1, device=next(model.parameters()).device))