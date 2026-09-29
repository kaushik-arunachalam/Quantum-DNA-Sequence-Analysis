"""Noisy simulators that approximate real NISQ hardware.

Every existing entry point (run_distribution, bbht_search, quantum_similarity)
already takes a `simulator=` argument, so these plug in without changes.
"""
from __future__ import annotations

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

BASIS = ["rz", "sx", "x", "cx"]  # typical superconducting-hardware basis


def build_noise_model(p1: float, p2: float, readout: float = 0.0) -> NoiseModel:
    """Depolarizing noise on 1- and 2-qubit gates plus symmetric readout error."""
    model = NoiseModel(basis_gates=BASIS)
    if p1 > 0:
        model.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["sx", "x", "rz"])
    if p2 > 0:
        model.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    if readout > 0:
        model.add_all_qubit_readout_error(
            ReadoutError([[1 - readout, readout], [readout, 1 - readout]]))
    return model


def noisy_simulator(p1: float, p2: float, readout: float = 0.0) -> AerSimulator:
    """Simulator that decomposes every gate into BASIS so the noise actually applies."""
    model = build_noise_model(p1, p2, readout)
    return AerSimulator(noise_model=model, basis_gates=model.basis_gates)
