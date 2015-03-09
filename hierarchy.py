from __future__ import division
from itertools import takewhile
from operator import itemgetter
import numpy as np
from dataset import read_labels


parent_to_child_mult = 0.4
child_to_parent_mult = 0.4
heat = 100

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
                echinopluteus
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
            SIPHONOPHORE_CALYCOPHORAN_ROCKETSHIP
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
    echinopluteus
APPENDICULARIAN
    appendicularian_fritillaridae
    appendicularian_s_shape
    appendicularian_slight_curve
    appendicularian_straight
UNKNOWN
    unknown_blobs_and_smudges
    unknown_sticks
unknown_unclassified
chordate_type1"""


def count_tabs(line):
    r"""Assumed that 4 spaces is a tab
    >>> count_tabs("no-tabs")
    0
    >>> count_tabs("        two-tabs-before")
    2
    """
    return (len(line) - len(line.lstrip())) // 4

rows = [(line.strip(), count_tabs(line)) for line in my_hierarchy.split('\n') if line]


class Node:
    def __init__(self, name, parent):
        self.children = []
        self.parent = parent
        self.name = name
        self.prob = None

    def __str__(self):
        return self.to_string()

    def to_string(self, prefix=''):
        if self.children:
            children = '\n'+'\n'.join(c.to_string(prefix+'  ') for c in self.children)
        else:
            children = ''
        p = '' if self.prob is None else ' (%.3f)' % self.prob

        return '%s%s%s%s' % (prefix, self.name, p, children)

    def is_root(self):
        return self.name == 'ROOT'

    def is_abstract(self):
        return self.name.upper() == self.name

    def map(self, f):
        f(self)
        for c in self.children:
            c.map(f)

    def reset_probs(self):
        self.prob = None
        for c in self.children:
            c.reset_probs()


def read_tree():
    root = Node('ROOT', None)
    last_nodes = {-1: root}
    for ln, cnt in rows:
        parent = last_nodes[cnt - 1]
        new_node = Node(ln, parent)
        parent.children.append(new_node)
        last_nodes[cnt] = new_node
        for i in range(cnt+1, max(last_nodes.keys())):
            if i in last_nodes:
                del last_nodes[i]
    return root


def find_node(name, root_of_tree):
    if root_of_tree.name == name:
        return root_of_tree
    for c in root_of_tree.children:
        found = find_node(name, c)
        if found:
            return found
    return None


def assign_probs(node, prob, parent_to_child_mult, child_to_parent_mult, override=True):
    if node.is_root():
        return
    if override or (node.prob is None):
        node.prob = prob
    else:
        return

    for c in node.children:
        assign_probs(c, prob * parent_to_child_mult, parent_to_child_mult, child_to_parent_mult, False)
    assign_probs(node.parent, prob * child_to_parent_mult, parent_to_child_mult, child_to_parent_mult, False)

# print(root)
# assign_probs(find_node('siphonophore_partial', root), 1.0)
# print(root)


def heated_targetings(label_to_int, train_y,
                      parent_to_child_mult=0.4, child_to_parent_mult=0.4, coldness=100):
    marginal_probs = train_y.sum(axis=0) / train_y.shape[0]
    tree = read_tree()
    probs_given_classes = []
    for lbl, yid in sorted(label_to_int.items(), key=itemgetter(1)):
        tree.reset_probs()
        assign_probs(find_node(lbl, tree), coldness, parent_to_child_mult, child_to_parent_mult)
        def assign_prior(node):
            if node.name not in label_to_int.keys():
                return
            marg_prob = marginal_probs[label_to_int[node.name]]
            if node.prob is None:
                node.prob = marg_prob
            else:
                node.prob = np.max(node.prob * marg_prob, marg_prob)
        tree.map(assign_prior)
        probs_given_class = []
        for lbl2, yid2 in sorted(label_to_int.items(), key=itemgetter(1)):
            node2 = find_node(lbl2, tree)
            if node2 is None:
                print('Not found %s!' % lbl2)
            probs_given_class.append(node2.prob)
        probs_given_classes.append(np.array(probs_given_class))
    probs_given_classes = np.vstack(probs_given_classes)
    probs_given_classes = probs_given_classes / probs_given_classes.sum(axis=1, keepdims=True)
    return np.cast['float32'](probs_given_classes[train_y.argmax(axis=1)])

# train_y = np.load('/media/marat/MySSD/plankton/train_bluntresize64_y.npy')
#
# unique_labels = read_labels()
# n_classes = len(unique_labels)
# label_to_int = {unique_labels[i]: i for i in range(n_classes)}
#
#
# soft_train_y = heated_targetings(label_to_int, train_y)
# print(-np.sum(train_y*np.log(soft_train_y), axis=1).mean())