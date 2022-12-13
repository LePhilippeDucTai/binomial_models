from binomial_trees import CRRModel, JRModel, BinomialTree

if __name__ == "__main__":
    r = 0.03
    sigma = 0.2
    dt = 1
    t_max = 5
    crr_model = CRRModel(r, sigma, t_max, dt)
    print(crr_model.lattice(1.0))

    bin_model = BinomialTree(r, sigma, t_max, dt)
    print(bin_model.lattice(1.0))
