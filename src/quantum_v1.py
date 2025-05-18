'''
QuantumV1 Agent Archetype
'''
import logging
import json
import os

from dotenv import load_dotenv
from qiskit import transpile
import qiskit.qasm2
import qiskit.qasm3
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime.fake_provider import FakeAlmadenV2
import numpy as np


from xmpp_agent import XmppAgent

logger = logging.getLogger("QuantumV1")

# Load env vars
load_dotenv()

IBM_CLOUD_API_KEY = os.getenv("IBM_CLOUD_API_KEY")
IBM_CLOUD_QISKIT_INSTANCE = os.getenv("IBM_CLOUD_QISKIT_INSTANCE")



class QuantumV1(XmppAgent):
  '''
  QuantumV1 provides chats with LLMs
  '''
  async def start(self):
    await super().start()
    try:
      pass
      # self.model = init_model(model_provider=self.config.options.model.provider,
      #                         model_name=self.config.options.model.name)
      # logger.debug(f"self.model: {self.model}")
    except Exception as e:
      logger.error(f"Error initializing model: {e}")

  async def chat(self, *, prompt, reply_func=None):
    try:
      logger.debug(f"prompt: {prompt}")
      logger.debug(f'self.config.options: {self.config.options}')
      qiskitConfig = self.config.options.qiskit
      logger.debug(f'qiskitConfig: {qiskitConfig}')

      service = QiskitRuntimeService(
        channel='ibm_cloud',
        instance=IBM_CLOUD_QISKIT_INSTANCE,
        token=IBM_CLOUD_API_KEY,
      )
      instances = service.instances()
      logger.debug(f'Instances: {instances}')
      backends = service.backends()
      logger.debug(f'Backends: {backends}')

      logger.debug(f'• Prompt: {prompt}')
      if qiskitConfig.language == "qasm2":
        circuit = qiskit.qasm2.loads(prompt)
      elif qiskitConfig.language == "qasm3":
        circuit = qiskit.qasm3.loads(prompt)
      logger.debug(f'Circuit:\n{circuit}')

      content = ""
      if qiskitConfig.draw.enable:
        drawing = circuit.draw(output=qiskitConfig.draw.output, style=qiskitConfig.draw.style)
        drawing = str(drawing)
        logger.debug('• Drawing: {drawing}')
        content += f'\n```qasm\n{drawing}\n```\n\n'

      # job = backend.run(transpiled_circuit)
      # counts = job.result().get_counts()
      # logger.debug(f'• Counts: {counts}')
      # output = json.dumps(counts, indent=2)
      # logger.debug(f'• Output: {output}')
      # content += output

      # backend = service.backend(qiskitConfig.backend)
      # logger.debug(f'Backend coupling_map: {backend.coupling_map}')
      # backend = service.least_busy(operational=True, simulator=False)
      # backend = FakeAlmadenV2()  # FIXME: Fake backend
      backend = service.least_busy(
        operational=True, simulator=False, min_num_qubits=127
      )
      logger.debug(f'Backend: {backend}')
      logger.debug(f'Backend coupling_map: {backend.coupling_map}')

      transpiled_circuit = transpile(circuit, backend)
      logger.debug(f'Transpiled circuit:\n{transpiled_circuit}')

      # # Set up six different observables.
      # observables_labels = ["IZ", "IX", "ZI", "XI", "ZZ", "XX"]
      # observables = [SparsePauliOp(label) for label in observables_labels]

      # # Convert to an ISA circuit and layout-mapped observables.
      # pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
      # isa_circuit = pm.run(transpiled_circuit)
      # # isa_circuit.draw("mpl", idle_wires=False)
      # isa_text = isa_circuit.draw("text", idle_wires=False)
      # logger.debug(f'isa_text:\n{isa_text}')
      # # Construct the Estimator instance.
      # estimator = Estimator(mode=backend)
      # estimator.options.resilience_level = 1
      # estimator.options.default_shots = 5000
      # mapped_observables = [
      #   observable.apply_layout(isa_circuit.layout) for observable in observables
      # ]
      # # One pub, with one circuit to run against five different observables.
      # job = estimator.run([(isa_circuit, mapped_observables)])
      # # Use the job ID to retrieve your job data later
      # print(f">>> Job ID: {job.job_id()}")

      # # Use the following code instead if you want to run on a simulator:
      # estimator = Estimator(backend)
      # # Convert to an ISA circuit and layout-mapped observables.
      # pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
      # isa_circuit = pm.run(transpiled_circuit)
      # mapped_observables = [
      #     observable.apply_layout(isa_circuit.layout) for observable in observables
      # ]
      # job = estimator.run([(isa_circuit, mapped_observables)])

      pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
      isa_circuit = pm.run(transpiled_circuit)
      print(f">>> Circuit ops (ISA): {isa_circuit.count_ops()}")
      sampler = Sampler(mode=backend)
      param_values = np.random.rand(circuit.num_parameters)  # The circuit is parametrized, so we will define the parameter values for execution
      job = sampler.run([(isa_circuit, param_values)])
      print(f">>> Job ID: {job.job_id()}")
      print(f">>> Job Status: {job.status()}")
      result = job.result()
      pub_result = result[0]
      logger.debug(f'pub_result: {pub_result}')
      logger.debug(f'pub_result.data: {pub_result.data}')
      logger.debug(f'result[0].data.c.get_counts(): {result[0].data.c.get_counts()}')
      # logger.debug(f'pub_result.data.meas.get_counts(): {pub_result.data.meas.get_counts()}')
      counts = result[0].data.c.get_counts()
      logger.debug(f'counts: {counts}')
      json_counts = json.dumps(counts, indent=2)
      content += f'\n```json\n{json_counts}\n```\n\n'

      # sampler = Sampler(backend)
      # job = sampler.run([transpiled_circuit])
      # print(f"job id: {job.job_id()}")
      # result = job.result()
      # logger.debug(f'• Result: {result}')
      # pub_result = result[0]
      # logger.debug(f'• Pub result: {pub_result}')
      # values = pub_result.data.evs
      # logger.debug(f'• Values: {values}')
      # errors = pub_result.data.stds
      # logger.debug(f'• Errors: {errors}')
      # counts = pub_result.data.meas.get_counts()
      # logger.debug(f'• Result counts: {counts}')
      # content += json.dumps(counts, indent=2)

      logger.debug(f'• Content: {content}')
      return content
    except Exception as e:
      logger.error(f"Quantum error: {e}")
      return f'Error: {str(e)}'

