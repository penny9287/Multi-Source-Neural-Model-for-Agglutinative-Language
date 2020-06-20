import tensorflow as tf
import opennmt as onmt

def model():
  return onmt.models.SequenceToSequence(
      source_inputter=onmt.inputters.ParallelInputter([
          onmt.inputters.WordEmbedder(
              vocabulary_file_key="source_vocabulary",
              embedding_size=512),
          onmt.inputters.ParallelInputter([
          onmt.inputters.WordEmbedder(
              vocabulary_file_key="stem_vocabulary",
              embedding_size=128),
          onmt.inputters.WordEmbedder(
              vocabulary_file_key="pos_vocabulary",
              embedding_size=32),
          onmt.inputters.WordEmbedder(
              vocabulary_file_key="suffix_vocabulary",
              embedding_size=64),
          onmt.inputters.WordEmbedder(
              vocabulary_file_key="tag_vocabulary",
              embedding_size=32)],
          reducer=onmt.layers.ConcatReducer())]),
      target_inputter=onmt.inputters.WordEmbedder(
          vocabulary_file_key="target_vocabulary",
          embedding_size=512),
      encoder=onmt.encoders.ParallelEncoder([
          onmt.encoders.BidirectionalRNNEncoder(
              num_layers=2,
              num_units=512,
              reducer=onmt.layers.ConcatReducer(),
              cell_class=tf.contrib.rnn.LSTMCell,
              dropout=0.3,
              residual_connections=False),
          onmt.encoders.BidirectionalRNNEncoder(
              num_layers=2,
              num_units=512,
              reducer=onmt.layers.ConcatReducer(),
              cell_class=tf.contrib.rnn.LSTMCell,
              dropout=0.3,
              residual_connections=False)],
          outputs_reducer=onmt.layers.ConcatReducer(axis=1)),
      decoder=onmt.decoders.AttentionalRNNDecoder(
          num_layers=4,
          num_units=512,
          bridge=onmt.layers.DenseBridge(),
          attention_mechanism_class=tf.contrib.seq2seq.LuongAttention,
          cell_class=tf.contrib.rnn.LSTMCell,
          dropout=0.3,
          residual_connections=False))
