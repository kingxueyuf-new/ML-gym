from keras.models import Model
from keras.layers import Input, LSTM, Dense

# Define an input sequence and process it.
num_encoder_tokens = 10
num_decoder_tokens = 10
latent_dim = 30


def build_model() -> Model():
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder_outputs, state_h, state_c = LSTM(units=latent_dim, return_state=True)(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_outputs, _, _ = LSTM(latent_dim, return_sequences=True, return_state=True)(decoder_inputs,
                                                                                       initial_state=encoder_states)
    decoder_outputs = Dense(num_decoder_tokens, activation='softmax')(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    return model


if __name__ == "__main__":
    model = build_model()
    encoder_input_data = None
    decoder_input_data = None
    decoder_target_data = None
    batch_size = 0
    epochs = 0
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2)
