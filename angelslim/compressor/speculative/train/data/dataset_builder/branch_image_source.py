"""The picture a .ckpt was generated from, ready for the branch forward.

Branch distillation re-scores the sequence with ONE token substituted. That
sequence was never generated, so unlike everything else an offline run consumes
it cannot be read off disk -- it needs a live target forward (see
``_OfflineBranchMixin`` in ...trainer/offline_eagle3_trainer.py). On a VLM that
forward needs the image.

tools/extract_images_beside_hidden.py writes each sample's source image next to
its .ckpt as ``data_<i>.img``, so finding it is an open() rather than a rebuild
of the source dataset -- which measured at 208s in every dataloader worker, and
got worse the more workers were added.

The image is processed at the processor's OWN default resolution. The branch
forward has to differ from the forward that produced the .ckpt in the
substituted token and in nothing else: the loss scores the teacher's logit
CHANGE between the two, so any second difference -- a coarser image included --
lands in that delta as if the token had caused it.
"""
import os

from transformers import AutoProcessor
from transformers.image_utils import load_image

_IMAGE_TOKEN_ENV = "BRANCH_DISTILL_IMAGE_TOKEN_ID"


class BranchImageSource:
    """Lazily built per dataloader worker; useless in the ones that never branch."""

    def __init__(self, target, image_token_id=None):
        if image_token_id is None:
            image_token_id = os.environ.get(_IMAGE_TOKEN_ENV)
        if not image_token_id:
            raise ValueError(
                f"offline branch distillation on a VLM needs {_IMAGE_TOKEN_ENV} so that a "
                "sample with image rows but no .img beside it is an error rather than a "
                "silent text-only re-score"
            )
        self.image_token_id = int(image_token_id)
        self.processor = AutoProcessor.from_pretrained(target)
        self.missing = 0

    def prepare(self, data, ckpt_path):
        """The vision inputs for this sample, or None when it has no image."""
        if not bool((data["input_ids"] == self.image_token_id).any()):
            return None  # text-only: the branch forward needs no picture
        path = ckpt_path.with_suffix(".img")
        if not path.exists():
            self.missing += 1
            if self.missing == 1:
                raise FileNotFoundError(
                    f"{path} is missing. Offline branch distillation re-scores the image "
                    f"along with the text, so it needs the source images beside the hidden "
                    f"states; run tools/extract_images_beside_hidden.py over this directory "
                    f"first. (Text-only samples legitimately have no .img, but this one has "
                    f"image rows in its input_ids.)"
                )
            return None
        processor = getattr(self.processor, "image_processor", self.processor)
        enc = processor(images=[load_image(str(path))], return_tensors="pt")
        keep = ("pixel_values", "pixel_attention_mask", "image_grid_thw")
        return {k: enc[k] for k in keep if k in enc}
