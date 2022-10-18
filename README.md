# household_classifier

## Classification Problem
As Computer Vision develops alongside the increasing prevalence of cheap edge computing devices, we are seeing more innovative CV use cases that can be deployed on relatively cheap edge devices in the home such as Ring doorbells. Most models focus on detection of an individual to start a recording or send a notification; this project intends to expand that list to include detection of:
- Pets
-- Cat
■ 4298 Images ○ Dog
■ 4562 Images
● Miscellaneous home items
○ Potted plant
■ 4624 Images
○ Backpack
■ 5756 Images
● Person
○ 66808 Images
This would allow models to expand their ability to recognize and interpret scenes while offering users additional functionality to monitor pet behavior and track the location of household items or the health of plants in their home by recognizing whether the image contains a certain category.

## [Dataset](cocodataset.org)
The COCO dataset comes in a variety of image sizes and resolutions. To standardize the
data, we’ll have to create a pipeline to enforce image size/resolutions and other preprocessing steps as necessary. We plan on utilizing histograms of oriented gradients and edge detection to identify objects within the image and textural features to aid in image classification across our categories.
