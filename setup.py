# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Attribution-NonCommercial 4.0 International
# License (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the
# License at

#     https://creativecommons.org/licenses/by-nc/4.0/

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import find_packages, setup

setup(
    name="protenix",
    version="1.0.0",
    description="",
    packages=find_packages(
        exclude=(
            "assets",
            "benchmark",
            "*.egg-info",
        )
    ),
    include_package_data=True,
    package_data={
        "protenix": ["model/layer_norm/kernel/*"],
    },
)