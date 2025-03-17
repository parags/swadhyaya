———————————————————————————————————
FOLDER STRUCTURE DESCRIPTION
———————————————————————————————————

- questionnaires: all there response questionnaires (Spanish); raw and preprocessed
Including SAM
|
——preprocessed: Ficha_Evaluacion_Participante_SAM_Refactored.csv: the SAM responses for every film clip


- key_moments: the key moment timestamps for every emotion’s clip

- empatica_wearable_data: XXXX
|
|—raw
|——1: ID = 1 of subject
|————empatica_slices: data from EMPATICA E4 device
|—————————ANGER_XXX.csv :  leg data of the anger elicitation
|—————————FEAR_XXX.csv :  leg data of the fear elicitation
|—————————HAPPINESS_XXX.csv :  leg data of the happiness elicitation
|—————————SADNESS_XXX.csv :  leg data of the sadness elicitation
|————order: film elicitation order of play: For example: HAPPINESS,SADNESS,ANGER,FEAR
…
|
|—preprocessed
|——unclean-signals: without artifacts, noise, etc.
|————empatica_slices: data from EMPATICA device
|—————————0.0078125: data resampled to 128 Hz
|——clean-signals: removed artifacts, noise, etc.
|————muse: EEG data of Muse device
|—————————0.0078125: data resampled to 128 Hz


