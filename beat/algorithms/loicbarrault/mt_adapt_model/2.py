# You may import any python packages that will be available in the environment you will run this algorithm in
# Environments can change based on the experiment's settings
import numpy
import pickle
import sidekit


class AlliesExtractor(sidekit.FeaturesExtractor):
    """
    A FeaturesExtractor process an audio file in SPHERE, WAVE or RAW PCM format and extract filter-banks,
    cepstral coefficients, bottle-neck features (in the future), log-energy and perform a speech activity detection.
    """
    def __init__(self,
                 dataloader,name_dict):
        super().__init__()
        self.dataloader = dataloader
        self.name_dict = name_dict
        self.audio_filename_structure = "{}"


    def extract(self,
                show, 
                channel, 
                input_audio_filename):
        # Create an HDF5 handler to return
        # If the output file name does not include the ID of the show,
        # (i.e., if the feature_filename_structure does not include {})
        # the feature_filename_structure is updated to use the output_feature_filename
        h5f = h5py.File('toto', 'a', backing_store=False, driver='core')
        # get features and vad from file (or from dataloader)
        #entry, _, _ = self.dataloader[self.name_dict[input_audio_filename]]
        entry, _, _ = self.dataloader[self.name_dict[show]]
        cep=entry["features"].value

        label = numpy.ones((cep.shape[0]), dtype=bool)
        sidekit.frontend.io.write_hdf5(show, h5f,
                   cep, cep.mean(1), cep.std(1),
                   None, None, None,
                   None, None, None,
                   None, None, None,
                   label,
                   'percentile')
        return h5f

def first_pass_segmentation(cep, show):
    thr_l = 2
    thr_h = 3
    thr_vit = -250
    # initialize diarization without VAD (VAD is be added later)
    sad_diar = init_seg(cep, show=input_file, cluster='init')
    #sad_diar = init_seg_from_label(input_file, label, frame_per_second=100)
    init_diar = copy.deepcopy(sad_diar)
    # ## Step 2: gaussian divergence segmentation
    seg_diar = segmentation.segmentation(cep, init_diar)
    # ## Step 3: linear BIC segmentation (fusion)
    bicl_diar = segmentation.bic_linear(cep, seg_diar, thr_l, sr=False)
    # ## Step 4: BIC HAC (clustering intra-show)
    hac = hac_bic.HAC_BIC(cep, bicl_diar, thr_h, sr=False)
    #print("start hac.perform")
    bich_diar = hac.perform()
    # Viterbi decoding
    vit_diar = viterbi.viterbi_decoding(cep, bich_diar, thr_vit) 
    return vit_diar

def iv_clustering(input_diar, model, features_server):
    idmap_in = input_diar.id_map()
    local_ivectors = model.train(features_server,idmap_in, normalization=False) # extract i-vectors on the current document

    tmp_ivectors = copy.deepcopy(local_ivectors) # not sure this line is useful

    tmp_ivectors.spectral_norm_stat1(model.norm_mean[:1], model.norm_cov[:1])
    ndx = sidekit.Ndx(models=tmp_ivectors.modelset, testsegs=tmp_ivectors.modelset)
    scores = fast_PLDA_scoring(tmp_ivectors, tmp_ivectors, ndx,
                               model.plda_mean,
                               model.plda_f,
                               model.plda_sigma,
                               p_known=0.0,
                               scaling_factor=1.0,
                               check_missing=False)
    scores.scoremat = 0.5 * (scores.scoremat + scores.scoremat.transpose())
    #
    # Do the clustering within-show

    output_diar, _, __ = hac_iv(input_diar, scores, threshold=-w_threshold)  

    return outputdiar


def adapt_plda(input_diar, model, features_server):
    # Il faudra supprimer les locuteurs qui n'ont pas assez de sessions
    idmap_in = input_diar.id_map()
    ivectors = model.train(features_server, idmap_in, normalization=False) # extract i-vectors on the current document

    # Normalize i-vectors and train PLDA
    norm_mean, norm_cov = ivectors.estimate_spectral_norm_stat1(1, 'sphNorm')

    # Train  PLDA
    plda_fa = sidekit.FactorAnalyser()

    plda_fa.plda(ivectors,
                 rank_f=20,
                 nb_iter=10,
                 scaling_factor=1.,
                 output_file_name=None,
                 save_partial=False)

    model.sn_mean = norm_mean
    model.sn_cov = norm_cov
    model.plda_mean = plda_fa.mean
    model.plda_f = plda_fa.F
    model.plda_g = plda_fa.G
    model.plda_sigma = plda_fa.Sigma

    return model


class Algorithm:
    # initialise fields to store cross-input data (e.g. machines, aggregations, etc.)
    def __init__(self):
        pass

    # this will be called each time the sync'd input has more data available to be processed
    def process(self, data_loaders, outputs):
        # Groups available:
        # Group 0:
        #   Input "model" with type  "system/text/1"
        #   Input "file_id" with type  "system/text/1"
        #   Input "speech" with type  "system/array_1d_floats/1"
        #   Input "speakers" with type  "allies/speakers/1"
        #   Input "uem" with type  "allies/uemranges/1"
        #   Output "model" with type  "system/text/1"
        #   Output "file_id" with type  "system/array_1d_text/1"
        #   Output "speakers" with type  "allies/speakers/1"

        # Create a Loader object to access all "inputs" from the previous blocks
        # Although this loader seems to focus on "features" is allows to access all "inputs"
        loader = data_loaders.loaderOf("features")

        

        # Get the model
        model_loader = data_loaders.loaderOf("model")
        model = pickle.loads(bytes(model_loader[0][0]['model'].text , "latin-1"))
        #import ipdb
        #ipdb.set_trace()

        # Fill a dictionnary to access features from the files
        name_dict = {}
        for i in range(loader.count()):
            file_id = loader[i][0]['file_info'].file_id
            name_dict[file_id] = int(i)

        # Create a sidekit.FeaturesServer object to load features from the platform
        fe = AlliesExtractor(loader,name_dict)
        fs = sidekit.FeaturesServer(features_extractor=fe, 
                                    dataset_list = ['cep'], 
                                    keep_all_features=True,
                                    delta=False,
                                    double_delta=False)

        # Here is the loop on files to process
        #   get the features
        #   get the UEM
        #   get the file_info
        for i in range(loader.count()):

            end = i
            (data, _, end) = loader[i]
            #uem = uem_loader[i][0]
            file_id = data['file_info'].file_id
            supervision = data['file_info'].supervision
            time_stamp = data['file_info'].time_stamp

            """
            Main diarization adaptation
            """

            # Compute the result to return (without system adaptation)
            spk = []
            st = []
            en = []
            for seg in model['global_diar']:
                #import ipdb
                #ipdb.set_trace()
                spk.append(seg[1])
                st.append(numpy.cast['float64'](seg[3])/100.)
                en.append(numpy.cast['float64'](seg[4])/100.)
            outputs['speakers'].write({ 'speaker': spk, 'start_time': st, 'end_time': en }, i)


            """
            cep = inputs['features'].data
            show = inputs['file_id'].data.text

            local_diar = first_pass_segmentation(cep, show)

            # Extract i-vectors within-show and perform within-show clustering
            iv_diar = iv_clustering(local_diar, self.model, self.fs)

            # Modify the cluster ID to add the show id as a prefix
            for seg in iv_diar.segments:
                seg['show'] = show + '_' + seg['show']

            self.global_diar.segments += iv_diar.segments

            # Perform cross show iv-clustering
            cross_diar = iv_clustering(global_diar, self.model, self.fs)

            # Si on adapte le modèle PLDA
            self.model = adapt_plda(input_diar, model, features_server)

            outputs.write({'model' : self.model})   # maqnque une serialisation
            outputs.write({'diarization' : cross_diar})  # A modifier pour utiliser les formats créés par Olivier

        # always return True, it signals BEAT to continue processing
        """
        model = pickle.dumps(model).decode('latin-1')
        outputs['model'].write({'text': model}, end)

        # always return True, it signals BEAT to continue processing
        return True







