import torch
import random
import time
import math

teacher_forcing_ratio = 0.5


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=10, device='cpu', SOS_token=0, EOS_token=1):
    print("TRAIN BATCH", input_tensor.shape, target_tensor.shape)
    # Initialize hidden layer
    encoder_hidden = encoder.initHidden(input_tensor.size(0))

    # Set gradients of all model parameters to zero
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Get batch size
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    print("SEQUENCE LENGTH", input_length, target_length)

    encoder_outputs = torch.zeros(
        max_length, encoder.hidden_size
    ).to(device)

    # Initialize loss
    loss = 0

    for i in range(input_length):
        print("I-TH TOKEN", i, input_tensor[i])
        # Pass input tensor through encoder -> get output tensor and hidden state tensor
        encoder_output, encoder_hidden = encoder(
            input_tensor[i], encoder_hidden)
        encoder_outputs[i] = encoder_output[0, 0]

    # Initialize the first token (SOS = Start of sequence) for the decoder
    decoder_input = torch.tensor([[SOS_token]], device=device)

    # The hidden state from the encoder is the initial hidden state of the decoder
    decoder_hidden = encoder_hidden

    # Determine if the decoder should use the true value for the next token in the sequence or its own prediction
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    # Compute gradient
    loss.backward()

    # Update weights of encoder and decoder
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, features, targets, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    # Initialize optimizer for encoder and decoder
    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate)

    # Initialize loss function
    criterion = torch.nn.NLLLoss()

    # Iterate over dataset
    for i in range(0, len(features)):
        input_tensor = features
        target_tensor = targets
        print(features.shape, targets.shape)

        loss = train(input_tensor[i], target_tensor[i], encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            #print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
