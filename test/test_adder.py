# source and inspiration from https://github.com/karpathy/minGPT/blob/master/projects/adder/adder.py
import numpy as np
from tensorli.models.transformerli import Transformerli
from tensorli.optimizerli import Adamli


class AdderDataset:
    def __init__(self, n_digits, split="train"):
        self.n_digits = n_digits
        self.split = split

        test_split_max = 0.1

        np.random.seed(4419)
        # each item is of shape (seq_len, 2), where the last column is the target
        assert self.n_digits <= 3, "only up to 3 digits supported otherwise too much memory used"
        all_possibilities = np.random.permutation((10**self.n_digits) ** 2)
        test_set = min(int(len(all_possibilities) * test_split_max), 500)  # max 500 test samples
        self.test = all_possibilities[:test_set]
        self.train = all_possibilities[test_set:]

    def __len__(self):
        return len(self.train) if self.split == "train" else len(self.test)

    def __getitem__(self, idx):
        if self.split == "train":
            x = self.train[idx % len(self.train)]
        else:
            x = self.test[idx % len(self.test)]
        division_factor = 10**self.n_digits
        a = x // division_factor
        b = x % division_factor
        c = a + b
        a_str = str(a).zfill(self.n_digits)
        b_str = str(b).zfill(self.n_digits)
        c_str = str(c).zfill(self.n_digits + 1)[::-1]
        complete_str = a_str + b_str + c_str
        assert len(complete_str) == 3 * self.n_digits + 1
        x = np.array([int(s) for s in complete_str])[:-1]
        y = np.array([int(s) for s in complete_str])[1:]
        y[: self.n_digits * 2 - 1] = -1
        return x, y


class DataSetli:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y


class DataLoaderli:
    def __init__(self, dataset: DataSetli, batch_size, random_sampling=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.random_sampling = random_sampling

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        if self.random_sampling:
            idxs = np.random.permutation(len(self.dataset))
        else:
            idxs = np.arange(len(self.dataset))
        for batch in range(len(self) // self.batch_size):
            x_batch = []
            y_batch = []
            for idx in idxs[batch * self.batch_size : (batch + 1) * self.batch_size]:
                x, y = self.dataset[idx]
                x_batch.append(x)
                y_batch.append(y)
            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)
            yield x_batch, y_batch


# deactivate test for ci as too much memory needed
def deactivated_test_adder_set():
    n_digits = 2
    adder_dataset_train = AdderDataset(n_digits=n_digits, split="train")
    adder_dataset_test = AdderDataset(n_digits=n_digits, split="test")

    vocab_size = 10
    seq_len = 2 + 2 + 2
    # GPT nano
    n_layer = 3
    n_head = 4
    n_embd = 64

    model = Transformerli(vocab_size, n_embd, seq_len, n_layer, n_head)

    batch_size = 64
    lr = 5e-4

    optimizer = Adamli(model.parameters(), lr=lr)

    train_dataset = DataSetli(adder_dataset_train)
    test_dataset = DataSetli(adder_dataset_test)

    train_loader = DataLoaderli(train_dataset, batch_size, random_sampling=True)

    loss_data = []

    def train_loop(epochs=5):
        for i in range(epochs):
            for bn, batch in enumerate(train_loader):
                x_batch, y_batch = batch
                logits = model(x_batch)

                out = logits.transpose(-1, -2)  # to fit pytorch cross entropy

                loss = out.cross_entropy(y_batch)
                print(f"Epoch {i}, batch {bn}: ", loss)
                loss_data.append(loss.data[0].item())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    train_loop(5)
    # loss data to csv
    loss_array = np.array(loss_data)
    np.savetxt("loss.csv", loss_array, delimiter=",")

    test_loader = DataLoaderli(test_dataset, batch_size, random_sampling=False)


    def eval_helper():
        total_correct = 0
        total = 0
        factors = np.array([10**i for i in range(n_digits + 1)][::-1])
        for batch in test_loader:
            x_batch, _ = batch
            x_only_digits = x_batch[:, : 2 * n_digits]

            digits_and_result = model.generate(x_only_digits, max_new_tokes=n_digits + 1)
            result = digits_and_result[:, -(n_digits + 1) :]
            result = result[:, ::-1]  # reverse the result
            result = (result * factors).sum(-1)

            first_digit = (x_only_digits[:, :n_digits] * factors[1:]).sum(-1)
            second_digit = (x_only_digits[:, n_digits : 2 * n_digits] * factors[1:]).sum(-1)
            print("first digit", first_digit, second_digit, result)
            prediction = result
            target = first_digit + second_digit
            correct = prediction == target
            print("correct", correct)
            count_correct = correct.sum()
            total_correct += count_correct
            total += len(x_batch)
        return total_correct, total

    total_correct, total = eval_helper()
    print("total correct", total_correct)
    print("total", total)
    print("accuracy", total_correct / total)

    return 0


if __name__ == "__main__":
    deactivated_test_adder_set()
