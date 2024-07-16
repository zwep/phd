Some changes need to be made to the provided config.yaml files. These include:

```
inference
    ...
 transforms:
      cropping:
        crop: null
        crop_type: uniform
        image_center_crop: true
      sensitivity_map_estimation:
        estimate_sensitivity_maps: true
        sensitivity_maps_gaussian: 0.7
      normalization:
        scaling_key: masked_kspace
```

And under 

```
validation
    ...
    crop_outer_slices: false
```

The `crop_outer_slices` is by default set to `true`. By setting it to `false` we avoid selecting some standard slices... why are not applicable to our case.



