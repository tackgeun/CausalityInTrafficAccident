# Details of dataset construction

## Download features
Download two RGB features extracted from i3d.
- [RGB](https://www.dropbox.com/s/s3b7r4cpbr6uqd5/i3d-rgb-fps25-Mar9th.pt?dl=0)
- [flipped-RGB](https://www.dropbox.com/s/0kiikl2yjco0xvn/i3d-rgb-flip-fps25-Mar9th.pt?dl=0)

## Statistics of dataset
### Semantic Taxonomy for Cause and Effect Events
<img width="480px" src="../figures/labels.png">

### Temporal Intervals
<img width="240px" src="../figures/cause_duration.png">
<img width="240px" src="../figures/effect_duration.png">

## The Other Details
Annotation tool
- We use and modify [BeaverDam](https://github.com/antingshen/BeaverDam) to annotate cause and effect event in an accident video.
- We modify BeaverDam to support both temporal regions and spatio-temporal regions of cause and effect event.
- But, we annotate videos with temporal localization due to an expensive annotation cost and the ambiguity of cause event of accident in spatio-temporal regions.
