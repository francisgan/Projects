```angular2html
env 1-4 are testing the increase in vcpus while scaling memory proportionally.
env 5-7 hold the number of vcpus constant at 4 and the cpu type constant, while scaling the memory to see its direct relationship.
env 8-10 hold the memory constant at 16 and the cpu type constant, while scaling the number of vcpus to see its direct relationship.
env 11-16 introduce different gpus to measure impact while attempting to hold vcpus and memory constant using available instance configurations.
```
| Environment | Instance     | vCPU | CPU Memory | CPU Type                 | GPU | GPU Memory | GPU Type         | done |
|-------------|--------------|------|------------|--------------------------|-----|------------|------------------|------|
| env1        | c7a.medium   | 1    | 2.0 GiB    | AMD EPYC Gen4            | 0   | N/A        | N/A              |3 trial|
| env2        | c7a.large    | 2    | 4.0 GiB    | AMD EPYC Gen4            | 0   | N/A        | N/A              |3 trial|
| env3        | c7a.xlarge   | 4    | 8.0 GiB    | AMD EPYC Gen4            | 0   | N/A        | N/A              |3 trial|
| env4        | c7a.2xlarge  | 8    | 16.0 GiB   | AMD EPYC Gen4            | 0   | N/A        | N/A              |3 trial|
| env5        | c6i.xlarge  | 4    | 8 GiB    | Intel Xeon 8375C (Ice Lake)  | 0   | N/A        | N/A              |3 trial|
| env6        | m6i.xlarge   | 4    | 16.0 GiB   | Intel Xeon 8375C (Ice Lake) | 0   | N/A        | N/A              |3 trial|
| env7        | r6i.xlarge   | 4    | 32.0 GiB   | Intel Xeon 8375C (Ice Lake) | 0   | N/A        | N/A              |3 trial|
| env8        | r6i.large  | 2    | 16.0 GiB    | Intel Xeon 8375C (Ice Lake)  | 0   | N/A        | N/A              |3 trial|
| env9        | m6i.xlarge   | 4    | 16.0 GiB   | Intel Xeon 8375C (Ice Lake) | 0   | N/A        | N/A              |3 trial|
| env10        | c6i.2xlarge   | 8    | 16.0 GiB   | Intel Xeon 8375C (Ice Lake) | 0   | N/A        | N/A              |3 trial|
| env11       | g3s.xlarge	  | 4    | 30.5	GiB   | Intel Xeon E5 2686 v4    | 1   | 8 GiB      | NVIDIA Tesla M60 |
| env12       | g4dn.xlarge  | 4    | 16 GiB     | Intel Cascade Lake       | 1   | 16 GiB     | NVIDIA T4        |
| env13       | g5.xlarge    | 4    | 16 GiB     | AMD EPYC Gen2            | 1   | 24 GiB     | NVIDIA A10G      |
| env14       | g3.4xlarge	  | 16   | 122 GiB    | Intel Xeon E5 2686 v4    | 1   | 8 GiB      | NVIDIA Tesla M60 |
| env15       | g4dn.4xlarge | 16   | 64 GiB     | Intel Cascade Lake       | 1   | 16 GiB     | NVIDIA T4        |
| env16       | g5.4xlarge   | 16   | 64 GiB     | AMD EPYC Gen2            | 1   | 24 GiB     | NVIDIA A10G      |
| datahub1       | N/A   | 1 core    | 32.0 GiB    | Intel(R) Xeon(R) Gold 5218          | 1   | N/A        | 2080ti        |1 trial|
| datahub2      | N/A   | 2 core    | 32.0 GiB    | Intel(R) Xeon(R) Gold 5218            | 1   | N/A        | 2080ti          |1 trial|
| datahub3       | N/A   | 4 core    | 32.0 GiB    | Intel(R) Xeon(R) Gold 5218             | 1   | N/A        | 2080ti        |1 trial|
| datahub4      | N/A   | 8 core    | 32.0 GiB    | Intel(R) Xeon(R) Gold 5218           | 1   | N/A        | 2080ti          |1 trial|
| datahub5       | N/A   | 12 core    | 32.0 GiB    | Intel(R) Xeon(R) Gold 5218             | 1   | N/A        | 2080ti        |1 trial|
| datahub6       | N/A   | 1 core    | 32.0 GiB    | AMD EPYC 7543P          | 1   | N/A        | a5000        |1 trial|
| datahub7      | N/A   | 2 core    | 32.0 GiB    | AMD EPYC 7543P              | 1   | N/A        | a5000          |1 trial|
| datahub8       | N/A   | 4 core    | 32.0 GiB    | AMD EPYC 7543P            | 1   | N/A        | a5000        |1 trial|
| datahub9      | N/A   | 8 core    | 32.0 GiB    | AMD EPYC 7543P           | 1   | N/A        | a5000          |1 trial|
| datahub10       | N/A   | 12 core    | 32.0 GiB    | AMD EPYC 7543P           | 1   | N/A        | a5000        |1 trial|
