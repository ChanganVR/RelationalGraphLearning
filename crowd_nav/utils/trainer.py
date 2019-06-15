import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class Trainer(object):
    def __init__(self, value_estimator, state_predictor, memory, device, batch_size, optimizer_str):
        """
        Train the trainable model of a policy
        """
        self.value_estimator = value_estimator
        self.state_predictor = state_predictor
        self.device = device
        self.criterion = nn.MSELoss().to(device)
        self.memory = memory
        self.data_loader = None
        self.batch_size = batch_size
        self.optimizer_str = optimizer_str
        self.v_optimizer = None
        self.s_optimizer = None
        self.pretend_batch_size = 100

    def set_learning_rate(self, learning_rate):
        if self.optimizer_str == 'Adam':
            self.v_optimizer = optim.Adam(self.value_estimator.parameters(), lr=learning_rate)
            self.s_optimizer = optim.Adam(self.state_predictor.parameters(), lr=learning_rate)
        elif self.optimizer_str == 'SGD':
            self.v_optimizer = optim.SGD(self.value_estimator.parameters(), lr=learning_rate, momentum=0.9)
            self.s_optimizer = optim.SGD(self.state_predictor.parameters(), lr=learning_rate)
        else:
            raise NotImplementedError
        logging.info('Lr: {} for parameters {} with {} optimizer'.format(learning_rate, ' '.join(
            [name for name, param in list(self.value_estimator.named_parameters()) +
             list(self.state_predictor.named_parameters())]), self.optimizer_str))

    def optimize_epoch_pretend_batch(self, num_epochs, writer):
        self.batch_size = 1

        if self.v_optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True, collate_fn=pack_batch)
        logging.info('start to optimize epoch in pretend batch manner')
        average_epoch_loss = 0
        for epoch in range(num_epochs):
            logging.debug('{}-th epoch starts'.format(epoch))
            epoch_loss = 0
            count_within_batch = 0

            values_list = []
            outputs_list = []
            for data in self.data_loader:
                if count_within_batch < self.pretend_batch_size:
                    input, value = data
                    self.v_optimizer.zero_grad()
                    output = self.value_estimator(input)
                    values_list.append(value)
                    outputs_list.append(output)
                    count_within_batch += 1
                else:
                    values = torch.cat(values_list, 0)
                    outputs = torch.cat(outputs_list, 0)
                    loss = self.criterion(outputs, values)
                    loss.backward()
                    self.v_optimizer.step()
                    epoch_loss += loss.data.item()
                    values_list = []
                    outputs_list = []
                    count_within_batch = 0
            logging.debug('{}-th epoch ends'.format(epoch))
            average_epoch_loss = epoch_loss / len(self.memory)
            writer.add_scalar('IL/average_epoch_loss', average_epoch_loss, epoch)
            logging.info('Average loss in epoch %d: %.2E', epoch, average_epoch_loss)
        return average_epoch_loss

    def optimize_epoch(self, num_epochs, writer):
        if self.v_optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        for epoch in range(num_epochs):
            epoch_v_loss = 0
            epoch_s_loss = 0
            logging.debug('{}-th epoch starts'.format(epoch))
            for data in self.data_loader:
                robot_states, human_states, values, next_human_states = data

                # optimize value estimator
                self.v_optimizer.zero_grad()
                outputs = self.value_estimator((robot_states, human_states))
                values = values.to(self.device)
                loss = self.criterion(outputs, values)
                loss.backward()
                self.v_optimizer.step()
                epoch_v_loss += loss.data.item()

                # optimize state predictor
                self.s_optimizer.zero_grad()
                _, next_human_states_est = self.state_predictor((robot_states, human_states), None)
                loss = self.criterion(next_human_states_est, next_human_states)
                loss.backward()
                self.s_optimizer.step()

                epoch_s_loss += loss.data.item()
            logging.debug('{}-th epoch ends'.format(epoch))
            writer.add_scalar('IL/epoch_v_loss', epoch_v_loss / len(self.memory), epoch)
            writer.add_scalar('IL/epoch_s_loss', epoch_s_loss / len(self.memory), epoch)
            logging.info('Average loss in epoch %d: %.2E, %.2E', epoch, epoch_v_loss / len(self.memory),
                         epoch_s_loss / len(self.memory))

        return
    
    def optimize_pretend_batch(self, num_batches):
        self.pretend_batch_size = 100
        self.batch_size = 1

        if self.v_optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True, collate_fn=pack_batch)
        logging.info('start to optimize:{} batches in pretend batch manner'.format(self.batch_size))
        losses = 0
        batch_count = 0
        count_within_batch = 0

        values_list = []
        outputs_list = []
        for data in self.data_loader:
            if count_within_batch < self.pretend_batch_size:
                input, value = data
                self.v_optimizer.zero_grad()
                output = self.value_estimator(input)
                values_list.append(value)
                outputs_list.append(output)
                count_within_batch += 1
            else:
                values = torch.cat(values_list, 0)
                outputs = torch.cat(outputs_list, 0)
                loss = self.criterion(outputs, values)
                loss.backward()
                self.v_optimizer.step()
                losses += loss.data.item()
                values_list = []
                outputs_list = []
                count_within_batch = 0
                batch_count += 1
            if batch_count > num_batches:
                break
        logging.info('end to optimize:{} batches in pretend batch manner'.format(self.batch_size))
        average_loss = losses / num_batches
        logging.info('Average loss : %.2E', average_loss)
        return average_loss

    def optimize_batch(self, num_batches):
        if self.v_optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        v_losses = 0
        s_losses = 0
        batch_count = 0
        for data in self.data_loader:
            robot_states, human_states, values, next_human_states = data

            # optimize value estimator
            self.v_optimizer.zero_grad()
            outputs = self.value_estimator((robot_states, human_states))
            values = values.to(self.device)
            loss = self.criterion(outputs, values)
            loss.backward()
            self.v_optimizer.step()
            v_losses += loss.data.item()

            # optimize state predictor
            self.s_optimizer.zero_grad()
            _, next_human_states_est = self.state_predictor((robot_states, human_states), None)
            loss = self.criterion(next_human_states_est, next_human_states)
            loss.backward()
            self.s_optimizer.step()
            s_losses += loss.data.item()

            batch_count += 1
            if batch_count > num_batches:
                break

        average_v_loss = v_losses / num_batches
        average_s_loss = s_losses / num_batches
        logging.info('Average loss : %.2E, %.2E', average_v_loss, average_s_loss)

        return average_v_loss, average_s_loss


def pad_batch(batch):
    """
    args:
        batch - list of (tensor, label)

    return:
        xs - a tensor of all examples in 'batch' after padding
        ys - a LongTensor of all labels in batch
    """
    # sort the sequences in the decreasing order of length
    sequences = sorted([x for x, y in batch], reverse=True, key=lambda x: x.size()[0])
    packed_sequences = torch.nn.utils.rnn.pack_sequence(sequences)
    xs = torch.nn.utils.rnn.pad_packed_sequence(packed_sequences, batch_first=True)
    ys = torch.Tensor([y for x, y in batch]).unsqueeze(1)

    return xs, ys


def pack_batch(batch):
    robot_states = torch.Tensor([x[0][0] for x, y in batch])
    human_states = torch.Tensor([x[1] for x, y in batch])
    values = torch.Tensor([y for y in batch])

    return (robot_states, human_states), values
