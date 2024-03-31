<h1> Enhancing Human Activity Recognition with Hardware Acceleration </h1>

<!-- Link to this website: https://louiecai.github.io/DSC-180B-Hardware-Acceleration/ -->

## Home

Welcome to our project page! This research focuses on improving Human Activity Recognition (HAR) through innovative hardware acceleration techniques.

## **About**

This project is focused on improving Human Activity Recognition (HAR) by using **specialized hardware**. Here's what we aim to achieve:

- **Boost Performance**: Making HAR systems <span style="color: #28a745;">faster and more efficient</span>.
- **Reduce Response Times**: Ensuring actions and responses are quicker.
- **Eliminate Bottlenecks**: Smoothing out any delays in the system.

Our work is important because HAR systems are a big part of many technologies we use every day, like:

- Wearable devices that monitor health
- Systems that help homes adapt to our needs

### WHat is Human Activity Recognition (HAR)?

Human Activity Recognition (HAR) is a technology that uses sensors to identify and classify human activities. It's used in a wide range of applications, from fitness trackers to smart homes. Here's a simple illustration of how it works:

<img src="network_illustration.png" alt="HAR Network" style="width: 100%; display: block;" />

<!-- explain how the HAR works -->

As the illustration shows, the HAR system uses **neural networks** to process data from sensors and identify human activities. These networks are trained to recognize patterns in the data, like the movements of a person's body.

The central issue we're addressing is that these networks can be very **_slow and inefficient_**, espcially on devices with limited processing power. Our project aims to make these systems faster and more efficient by leveraging hardware acceleration.

### Why This Matters

- **Better Devices**: Enhancing performance and user experience in wearable tech.
- **Healthcare Advances**: Offering real-time health monitoring, <span style="color: #dc3545;">potentially saving lives</span>.
- **Smarter Homes**: Creating responsive environments that adjust to what people need.

Through our project, we're not just improving technology. We're working to make it blend more seamlessly into our lives, enhancing every aspect of daily living.

<details>
<summary><b>More on Our Approach</b></summary>
<p>In diving deeper into our project:</p>
<ul>
  <li>We're moving past traditional software optimization to leverage <span style="color: #17a2b8;">hardware acceleration</span>.</li>
  <li>This shift addresses the current inefficiencies in processing HAR tasks, promising substantial improvements in system performance and energy usage.</li>
  <li>By tackling these core challenges, we anticipate our solutions will be more agile, reliable, and suited to their intended uses, significantly improving operational efficiency and the quality of life.</li>
</ul>
<p>Our efforts aim to impact various fields significantly, including wearable technology, healthcare, and intelligent home systems, marking a significant stride toward a future where technology enhances every aspect of our lives in real time.</p>

</details>

## Methodology

### Development Setup

The data collection process for our project is meticulously designed to evaluate the performance of Human Activity Recognition (HAR) neural networks across a spectrum of hardware configurations. Key components of this process include:

- **AWS Setup**: We leverage AWS instances to provide the computational power needed for our experiments. Each instance is tailored to match the requirements of different HAR neural network configurations.
- **Docker Containers**: To ensure consistency and reproducibility, Docker containers are deployed. These containers are pre-configured with all necessary dependencies to facilitate the running of our models across various environments.
- **Execution Scripts**: Automation scripts are crucial for streamlining the data collection process. These scripts orchestrate the setup, execution, and data gathering across different hardware setups, ensuring efficiency and accuracy in our findings.

<details>
<summary><b>More on Hardware Configurations</b></summary>

<p>In the context of our research, we explore a diverse array of hardware configurations to understand their impact on HAR system performance. This exploration includes but is not limited to:</p>
<ul>
    <li>Low-end CPUs that mirror the processing capabilities of wearable tech devices.</li>
    <li>Multi-core CPUs to assess the benefits of parallel processing.</li>
    <li>GPUs with varying tensor core counts to evaluate acceleration benefits for neural network computations.</li>
</ul>
<p>The data collection methodology is thorough, involving steps to ensure that each model is evaluated under identical conditions across the hardware spectrum. Post-collection, data is processed and analyzed to draw insights into performance variations, efficiency gains, and potential bottlenecks in HAR applications. This comprehensive approach allows us to pinpoint optimal hardware configurations that balance computational power, energy consumption, and real-world applicability.</p>

<p>Here are some examples of the hardware configurations we explored:</p>

<table>
    <thead>
        <tr>
            <th>Environment</th>
            <th>Instance</th>
            <th>vCPU</th>
            <th>Memory (GiB)</th>
            <th>CPU Type</th>
            <th>GPU</th>
            <th>GPU Memory (GiB)</th>
            <th>GPU Type</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>env1</td>
            <td>c7a.medium</td>
            <td>1</td>
            <td>2.0</td>
            <td>AMD EPYC Gen4</td>
            <td>0</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>env2</td>
            <td>c7a.large</td>
            <td>2</td>
            <td>4.0</td>
            <td>AMD EPYC Gen4</td>
            <td>0</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>env3</td>
            <td>c7a.xlarge</td>
            <td>4</td>
            <td>8.0</td>
            <td>AMD EPYC Gen4</td>
            <td>0</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>...</td>
            <td>...</td>
            <td>...</td>
            <td>...</td>
            <td>...</td>
            <td>...</td>
            <td>...</td>
            <td>...</td>
        </tr>
        <tr>
            <td>env16</td>
            <td>g5.4xlarge</td>
            <td>16</td>
            <td>64.0</td>
            <td>AMD EPYC Gen2</td>
            <td>1</td>
            <td>24.0</td>
            <td>NVIDIA A10G</td>
        </tr>
        <tr>
            <td>datahub1</td>
            <td>-</td>
            <td>1</td>
            <td>32.0</td>
            <td>Intel(R) Xeon(R) Gold 5218</td>
            <td>1</td>
            <td>-</td>
            <td>2080ti</td>
        </tr>
        <tr>
            <td>datahub10</td>
            <td>-</td>
            <td>12</td>
            <td>32.0</td>
            <td>AMD EPYC 7543P</td>
            <td>1</td>
            <td>-</td>
            <td>a5000</td>
        </tr>
    </tbody>
</table>

<p>Note: "-" indicates that the system does not include a GPU or the specific information is not applicable.</p>

</details>

### Data Collection

The networks are executed on the target hardware environments, and the execution traces are recorded. The data is then processed and analyzed to draw insights into performance variations, efficiency gains, and potential bottlenecks in HAR applications.

Here is an example of what the collected execution trace looks like:

<img src="execution_trace.png" alt="Execution Trace" style="width: 100%; display: block;" />

This execution trace is then processed as numbers and analyzed to draw insights into performance variations, efficiency gains, and potential bottlenecks in HAR applications.

## Results

<div style="display: grid; place-items: center; min-height: 100vh;">

<iframe src="runtime_cpu.html" width="90%" frameborder="0" height="500"></iframe>

<iframe src="runtime_vcpu.html" width="90%" frameborder="0" height="500"></iframe>

</div>

## Conclusion
