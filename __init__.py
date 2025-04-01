from .LayerAnimate_nodes import LoadImages, LoadPretrainedModel, LayerAnimateNode

NODE_CLASS_MAPPINGS = {
    "LoadImages": LoadImages,
    "LoadPretrainedModel": LoadPretrainedModel,
    "LayerAnimateNode": LayerAnimateNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImages": "Load Images",
    "LoadPretrainedModel": "Load Pretrained Model",
    "LayerAnimateNode": "Layer Animate Process",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
