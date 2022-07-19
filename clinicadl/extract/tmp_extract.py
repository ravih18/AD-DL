def pet_linear_nii(acq_label, suvr_reference_region, uncropped_image, use_uniform):
    import os

    if uncropped_image:
        description = ""
    else:
        description = "_desc-Crop"

    if use_uniform:
        description2 = "Uniform PET data"
        rec = "_rec-uniform"
    else:
        description2 = ""
        rec = ""

    information = {
        "pattern": os.path.join(
            "pet_linear",
            f"*_trc-{acq_label}{rec}_pet_space-MNI152NLin2009cSym{description}_res-1x1x1_suvr-{suvr_reference_region}_pet.nii.gz",
        ),
        "description": description2,
        "needed_pipeline": "pet-linear",
    }
    return information
