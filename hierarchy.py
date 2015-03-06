hierarchy = """ARTIFACTS
    artifacts_edge
    artifacts
PLANKTON
    FISH
        fish_larvae_myctophids
        fish_larvae_medium_body
        fish_larvae_deep_body
        fish_larvae_leptocephali
        fish_larvae_very_thin_body
        fish_larvae_thin_body
    GELATINOUS ZOOPLANKTON
        jellies_tentacles
        PELAGIC TUNICATES
            tunicate_doliolid
                tunicate_doliolid_nurse
            tunicate_salp
                tunicate_salp_chains
            tunicate_partial
        SIPHONOPHORES
            siphonophore_partial
            siphonophore_other_parts
            siphonophore_physonect
                siphonophore_physonect_young
            CALYCOPHORAN SIPHONOPHORES
                siphonophore_calycophoran_abylidae
                siphonophore_calycophoran_sphaeronectes
                    siphonophore_calycophoran_sphaeronectes_young
                    siphonophore_calycophoran_sphaeronectes_stem
                siphonophore_calycophoran_rocketship
                    siphonophore_calycophoran_rocketship_young
                    siphonophore_calycophoran_rocketship_adult
        ephyra
        CTENOPHORES
            ctenophore_cestid
            ctenophore_cydippid_no_tentacles
            ctenophore_cydippid_tentacles
            ctenophore_lobate
        HYDROMEDUSAE
            hydromedusae_narcomedusae
                hydromedusae_narco_dark
                hydromedusae_solmundella
                hydromedusae_solmaris
                    hydromedusae_narco_young
            hydromedusae_aglaura
            hydromedusae_liriope
            hydromedusae_haliscera
                hydromedusae_haliscera_small_sideview
            OTHER HYDROMEDUSAE
                PARTIAL
                    hydromedusae_partial_dark
                hydromedusae_bell_and_tentacles
                hydromedusae_typeD_bell_and_tentacles
                    hydromedusae_typeD
                hydromedusae_shapeA
                    hydromedusae_shapeA_sideview_small
                    hydromedusae_sideview_big
                hydromedusae_h15
                hydromedusae_shapeB
                hydromedusae_typeE
                hydromedusae_typeF
                hydromedusae_other
    DIATOMS
        diatom_chain_string
        diatom_chain_tube
    TRICHODESMIUM
        trichodesmium_tuft
        trichodesmium_bowtie
        trichodesmium_puff
        trichodesmium_multiple
    PROTISTS
        acantharia_protist
            acantharia_protist_big_center
            acantharia_protist_halo
        protist_noctiluca
        protist_other
            protist_star
            protist_fuzzy_olive
            protist_dark_center
        radiolarian_colony
            radiolarian_chain
    CRUSTACEANS
        crustacean_other
        COPEPODS
            CYCLOPOID COPEPODS
                copepod_cyclopoid_copilia
                copepod_cyclopoid_oithona
                    copepod_cyclopoid_oithona_eggs
            copepod_calanoid
                copepod_other
                copepod_calanoid_eucalanus
                copepod_calanoid_large
                    copepod_calanoid_large_side_antennatucked
                copepod_calanoid_octomoms
                copepod_calanoid_eggs
                copepod_calanoid_flatheads
                copepod_calanoid_frillyAntennae
                copepod_calanoid_small_longantennae
        stomatopod
        amphipods
        SHRIMP-LIKE
            shrimp-like_other
            euphausiids
                euphausiids_young
            decapods
                shrimp_zoea
                shrimp_caridean
                shrimp_sergestidae
    CHAETOGNATHS
        CHAETOGNATH
            chaetognath_other
            chaetognath_non_sagitta
            chaetognath_sagitta
        polychaete
    GASTROPODS
        heteropod
        PTEROPODS
            pteropod_butterfly
            pteropod_theco_dev_seq
            pteropod_triangle
    DETRITUS
        fecal_pellet
        detritus_blob
        detritus_filamentous
        detritus_other
    OTHER INVERT LARVAE
        invertebrate_larvae_other_A
        invertebrate_larvae_other_B
        trochophore_larvae
        tornaria_acorn_worm_larvae
        ECHINODERMS
            echinoderm_larva_seastar_bipinnaria
                echinoderm_larva_seastar_brachiolaria
            echinoderm_seacucumber_auricularia_larva
            ECHINODERM
                echinoderm_larva_pluteus_brittlestar
                echinoderm_larva_pluteus_early
                echinoderm_larva_pluteus_typeC
                echinoderm_larva_pluteus_urchin
    APPENDICULARIAN
        appendicularian_fritillaridae
        appendicularian_s_shape
        appendicularian_slight_curve
        appendicularian_straight
    UNKNOWN
        unknown_blobs_and_smudges
        unknown_sticks
        unknown_unclassified
    chordate_type1
"""
similarities = [
    ['acantharia_protist', 'protist_star'],
    ['protist_noctiluca', 'protist_fuzzy_olive'],
    ['hydromedusae_typeD_bell_and_tentacles', 'hydromedusae_bell_and_tentacles'],
    ['hydromedusae_h15', 'hydromedusae_shapeB'],
    ['siphonophore_calycophoran_rocketship_young',
     'siphonophore_calycophoran_sphaeronectes_young',
     'siphonophore_calycophoran_sphaeronectes_stem'],
    ['appendicularian_s_shape',
     'appendicularian_slight_curve',
     'appendicularian_straight'],
    ['shrimp_zoea',
     'shrimp_caridean'],
    ['chaetognath_other', 'chaetognath_non_sagitta'],
    ['echinopluteus',
     'echinoderm_larva_pluteus_brittlestar',
     'echinoderm_larva_pluteus_early',
     'echinoderm_larva_pluteus_urchin']
]


my_hierarchy = """ARTIFACTS
    artifacts_edge
    artifacts
FISH
    fish_larvae_myctophids
    fish_larvae_medium_body
    fish_larvae_deep_body
    fish_larvae_leptocephali
    fish_larvae_very_thin_body
    fish_larvae_thin_body
ephyra
ctenophore_cestid
ctenophore_lobate
hydromedusae_typeF
GELATINOUS ZOOPLANKTON
    jellies_tentacles
    PELAGIC TUNICATES
        tunicate_doliolid
            tunicate_doliolid_nurse
        tunicate_salp
            tunicate_salp_chains
        tunicate_partial
    SIPHONOPHORES
        siphonophore_partial
        siphonophore_other_parts
        siphonophore_physonect
            siphonophore_physonect_young
        CALYCOPHORAN SIPHONOPHORES
            siphonophore_calycophoran_abylidae
                echinoderm_seacucumber_auricularia_larva
            siphonophore_calycophoran_sphaeronectes
                siphonophore_calycophoran_sphaeronectes_young
                siphonophore_calycophoran_sphaeronectes_stem
            siphonophore_calycophoran_rocketship
                siphonophore_calycophoran_rocketship_young
                siphonophore_calycophoran_rocketship_adult
        ctenophore_cydippid_no_tentacles
        ctenophore_cydippid_tentacles
    HYDROMEDUSAE
        hydromedusae_narcomedusae
            hydromedusae_narco_dark
            hydromedusae_solmundella
            hydromedusae_solmaris
                hydromedusae_narco_young
        hydromedusae_aglaura
        hydromedusae_liriope
        hydromedusae_haliscera
            hydromedusae_haliscera_small_sideview
        OTHER HYDROMEDUSAE
            PARTIAL
                hydromedusae_partial_dark
            hydromedusae_bell_and_tentacles
            hydromedusae_typeD_bell_and_tentacles
                hydromedusae_typeD
            hydromedusae_shapeA
                hydromedusae_shapeA_sideview_small
                hydromedusae_sideview_big
            hydromedusae_h15
            hydromedusae_shapeB
            hydromedusae_typeE
            hydromedusae_other
DIATOMS
    diatom_chain_string
    diatom_chain_tube
TRICHODESMIUM
    trichodesmium_tuft
    trichodesmium_bowtie
    trichodesmium_puff
    trichodesmium_multiple
PROTISTS
    acantharia_protist
        acantharia_protist_big_center
        acantharia_protist_halo
    protist_noctiluca
    protist_other
        protist_star
        protist_fuzzy_olive
        protist_dark_center
radiolarian_colony
    radiolarian_chain
CRUSTACEANS
    crustacean_other
    COPEPODS
        CYCLOPOID COPEPODS
            copepod_cyclopoid_copilia
            copepod_cyclopoid_oithona
                copepod_cyclopoid_oithona_eggs
        copepod_calanoid
            copepod_other
            copepod_calanoid_eucalanus
            copepod_calanoid_large
                copepod_calanoid_large_side_antennatucked
            copepod_calanoid_octomoms
            copepod_calanoid_eggs
            copepod_calanoid_flatheads
            copepod_calanoid_frillyAntennae
            copepod_calanoid_small_longantennae
    stomatopod
    amphipods
    SHRIMP-LIKE
        shrimp-like_other
        euphausiids
            euphausiids_young
        decapods
            shrimp_zoea
            shrimp_caridean
            shrimp_sergestidae
CHAETOGNATHS
    CHAETOGNATH
        chaetognath_other
        chaetognath_non_sagitta
        chaetognath_sagitta
    polychaete
GASTROPODS
    heteropod
    PTEROPODS
        pteropod_butterfly
        pteropod_theco_dev_seq
pteropod_triangle
DETRITUS
    fecal_pellet
    detritus_blob
    detritus_filamentous
    detritus_other
invertebrate_larvae_other_A
invertebrate_larvae_other_B
trochophore_larvae
tornaria_acorn_worm_larvae
ECHINODERMS
    echinoderm_larva_seastar_bipinnaria
        echinoderm_larva_seastar_brachiolaria
ECHINODERM
    echinoderm_larva_pluteus_brittlestar
    echinoderm_larva_pluteus_early
    echinoderm_larva_pluteus_typeC
    echinoderm_larva_pluteus_urchin
APPENDICULARIAN
    appendicularian_fritillaridae
    appendicularian_s_shape
    appendicularian_slight_curve
    appendicularian_straight
UNKNOWN
    unknown_blobs_and_smudges
    unknown_sticks
unknown_unclassified
chordate_type1
"""
