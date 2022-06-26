const Nou = (function() {

function fillArray(length, value = 0) {
    const arr = new Array(length);
    for(let i = 0; i < length; i += 1) { arr[i] = value; }
    return arr;
}
function findVolume(dimensions) {
    let volume = 1;
    for(let i = 0; i < dimensions.length; i += 1) {
        volume *= dimensions[i] || 1;
    }
    return volume;
}

// The vectors have gradients
const Vector = (function() {
    const proto = {
        index(a) {
            let index = a[this.shape.length - 1];
            for(let i = this.shape.length - 2; i >= 0; i -= 1) {
                index = a[i] + this.shape[i] * index;
            }
            return index;
        },
        position(index) {
            const position = Array(this.shape.length);
            for(let i = 0; i < this.shape.length; i += 1) {
                if(index > 0) {
                    position[i] = (index - this.shape[i] * ((index / this.shape[i]) | 0));
                    index = (index - position[i]) / this.shape[i];
                }
                else { position[i] = 0; }
            }
            return position;
        },
        clone() {
            return Vector(this.shape, this.values);
        },
    };
    function Vector(dimensions, values) {
        const vector = Object.create(proto);
        vector.shape = dimensions;
        let volume = findVolume(dimensions);
        if(typeof values === "number") {
            vector.values = fillArray(volume, values);
        }
        else if(!values && values !== 0) {
            vector.values = new Float64Array(volume);
            const s = 1 / Math.sqrt(findVolume(dimensions));
            for(let i = 0; i < volume; i += 1) {
                vector.values[i] = Math.random() * s * 2 - s;
            }
        }
        else {
            vector.values = new Float64Array(values);
        }
        vector.gradient = new Float64Array(volume).fill(0);
        return vector;
    };
    return Vector;
})();

const Dense = (function() {
    const proto = {
        type: "dense",
        forward() {
            const out = this.out;
            const inValues = this.in.values;
            for(let i = 0; i < out.values.length; i += 1) {
                out.values[i] = this.biases.values[i];
                for(let j = 0; j < inValues.length; j += 1) {
                    out.values[i] += this.weights[i].values[j] * inValues[j];
                }
            }
            return out;
        },
        backward() {
            const input = this.in, out = this.out;
            for(let i = 0; i < out.values.length; i += 1) {
                const weights = this.weights[i], derivative = out.gradient[i]; // dCost/dActivation
                for(let j = 0; j < input.values.length; j += 1) {
                    weights.gradient[j] += input.values[j] * derivative; // dCost/dWeight = dActivation/dWeight * dCost/dActivation
                    input.gradient[j] += weights.values[j] * derivative; // dCost/dInput = dActivation/dInput * dCost/dActivation
                }
                this.biases.gradient[i] += derivative; // dCost/dBias = dActivation/dBias * dCost/dActivation
            }
        },
        get parameters() { return [...this.weights, this.biases]; },
    };
    function Dense(properties) {
        const { neurons } = properties;
        const layer = addLayer(this, proto);
        layer.outShape = neurons;
        layer.out = Vector(layer.outShape, 0);
        layer.weights = Array(findVolume(neurons));
        for(let i = 0; i < layer.weights.length; i += 1) {
            layer.weights[i] = Vector(layer.inShape);
        }
        layer.biases = Vector(neurons);
        return layer;
    }
    return Dense;
})();

const Convolution = (function() {
    const proto = {
        type: "convolution",
        forward() {
            const dimensionality = this.kernelSize.length;
            const out = this.out;
            const filterPosition = [];
            let filterDepthMultiplier = 1;
            for(let i = 0; i < dimensionality; i += 1) {
                filterPosition[i] = -this.padding[i];
                filterDepthMultiplier *= this.kernelSize[i];
            }
            const outPosition = [];
            for(let i = 0; i < dimensionality + 1; i += 1) {
                outPosition[i] = 0;
            }
            const filterVolume = findVolume(this.kernelSize);
            let inDepthMultiplier = 1;
            const inputDimensionIncrements = [];
            for(let k = 0; k < dimensionality; k += 1) {
                inDepthMultiplier *= this.inShape[k];
                inputDimensionIncrements[k] = inDepthMultiplier;
            }

            // Feature maps
            for(let i = 0; i < out.values.length; i += 1) {
                let activation = 0;
                const filterIndex = outPosition[dimensionality];
                const filter = this.filters[filterIndex];
                const filterPixel = [];
                let inIndex = 0;
                for(let j = dimensionality - 1; j >= 0; j -= 1) {
                    filterPixel[j] = 0;
                    inIndex = filterPosition[j] + this.inShape[j] * inIndex;
                }
                let startIndex = inIndex;
                for(let j = 0; j < filterVolume; j += 1) {
                    for(let k = 0; k < this.inShape[dimensionality]; k += 1) {
                        // Input pixel with depth k and lth coordinate (filterPosition[l] + filterPixel[l] * dilation[l])
                        const index = inIndex + k * inDepthMultiplier;
                        activation += (this.in.values[index] || 0) * filter.values[j];
                    }

                    // Increment pixel of filter
                    filterPixel[0] += 1;
                    inIndex += this.dilation[0];
                    for(let k = 0; k < dimensionality - 1; k += 1) {
                        if(filterPixel[k] >= this.kernelSize[k]) {
                            for(let m = 0; m <= k; m += 1) {
                                filterPixel[m] = 0;
                            }
                            startIndex = inIndex = startIndex + inputDimensionIncrements[k];
                            filterPixel[k + 1] += 1;
                        }
                    }
                }
                out.values[i] = activation + this.biases.values[filterIndex];
                // out.values[i] = activation;

                // Move filter
                filterPosition[0] += this.stride[0];
                outPosition[0] += 1;
                for(let j = 0; j < dimensionality; j += 1) {
                    if(outPosition[j] >= this.outShape[j]) {
                        for(let k = 0; k <= j; k += 1) {
                            outPosition[k] = 0;
                            filterPosition[k] = -this.padding[k];
                        }
                        if(j < dimensionality - 1) { filterPosition[j + 1] += this.stride[j + 1]; }
                        outPosition[j + 1] += 1;
                    }
                }
            }
            return out;
        },
        backward() {
            const dimensionality = this.kernelSize.length;
            const out = this.out;
            const filterPosition = [];
            let filterDepthMultiplier = 1;
            for(let i = 0; i < dimensionality; i += 1) {
                filterPosition[i] = -this.padding[i];
                filterDepthMultiplier *= this.kernelSize[i];
            }
            const outPosition = [];
            for(let i = 0; i < dimensionality + 1; i += 1) {
                outPosition[i] = 0;
            }
            const filterVolume = findVolume(this.kernelSize);
            let inDepthMultiplier = 1;
            const inputDimensionIncrements = [];
            for(let k = 0; k < dimensionality; k += 1) {
                inDepthMultiplier *= this.inShape[k];
                inputDimensionIncrements[k] = inDepthMultiplier;
            }

            // Feature maps
            for(let i = 0; i < out.values.length; i += 1) {
                const derivative = this.out.gradient[i]; // dCost/dActivation
                const filterIndex = outPosition[dimensionality];
                const filter = this.filters[filterIndex];
                const filterPixel = [];
                let inIndex = 0;
                for(let j = dimensionality - 1; j >= 0; j -= 1) {
                    filterPixel[j] = 0;
                    inIndex = filterPosition[j] + this.inShape[j] * inIndex;
                }
                let startIndex = inIndex;
                for(let j = 0; j < filterVolume; j += 1) {
                    for(let k = 0; k < this.inShape[dimensionality]; k += 1) {
                        // Input pixel with depth k and lth coordinate (filterPosition[l] + filterPixel[l] * dilation[l])
                        const index = inIndex + k * inDepthMultiplier;
                        filter.gradient[j + k * filterDepthMultiplier] += this.in.values[index] * derivative; // dCost/dFilter = dActivation/dFilter * dCost/dActivation
                        this.in.gradient[index] += filter.values[j + k * filterDepthMultiplier] * derivative; // dCost/dInput = dActivation/dInput * dCost/dActivation
                    }

                    // Increment pixel of filter
                    filterPixel[0] += 1;
                    inIndex += this.dilation[0];
                    for(let k = 0; k < dimensionality - 1; k += 1) {
                        if(filterPixel[k] >= this.kernelSize[k]) {
                            for(let m = 0; m <= k; m += 1) {
                                filterPixel[m] = 0;
                            }
                            startIndex = inIndex = startIndex + inputDimensionIncrements[k];
                            filterPixel[k + 1] += 1;
                        }
                    }
                }
                this.biases.gradient[filterIndex] += derivative; // dCost/dBias = dActivation/dBias * dCost/dActivation

                // Move filter
                filterPosition[0] += this.stride[0];
                outPosition[0] += 1;
                for(let j = 0; j < dimensionality; j += 1) {
                    if(outPosition[j] >= this.outShape[j]) {
                        for(let k = 0; k <= j; k += 1) {
                            outPosition[k] = 0;
                            filterPosition[k] = -this.padding[k];
                        }
                        if(j < dimensionality - 1) { filterPosition[j + 1] += this.stride[j + 1]; }
                        outPosition[j + 1] += 1;
                    }
                }
            }
        },
        get parameters() { return [...this.filters, this.biases]; },
    };
    function Convolution(properties) {
        const layer = addLayer(this, proto);
        const dimensionality = layer.inShape.length - 1;
        const { filters = 1, kernelSize = Array(dimensionality).fill(1), stride = Array(dimensionality).fill(1), dilation = Array(dimensionality).fill(1), padding = Array(dimensionality).fill(0) } = properties;
        layer.kernelSize = kernelSize;
        layer.stride = stride;
        layer.dilation = dilation;
        layer.padding = padding;
        layer.filters = Array(filters);
        for(let i = 0; i < filters; i += 1) {
            layer.filters[i] = Vector(kernelSize);
        }
        layer.biases = Vector([filters]);
        const out = [];
        for(let i = 0; i < dimensionality; i += 1) {
            const size = (layer.inShape[i] + padding[i] * 2 - kernelSize[i] * dilation[i]) / stride[i];
            if(size !== (size | 0)) { console.warn("Convolution input not divisible with current parameters; filters will not convolve across all values."); }
            out[i] = size + 1;
        }
        out[out.length] = filters;
        layer.outShape = out;
        layer.out = Vector(layer.outShape, 0);
        return layer;
    }
    return Convolution;
})();

const RelU = (function() {
    const proto = {
        type: "relu",
        forward() {
            const input = this.in.values, out = this.out.values;
            for(let i = 0; i < out.length; i += 1) {
                const v = input[i];
                out[i] = v > 0 ? v : 0;
            }
            return this.out;
        },
        backward() {
            const input = this.in.gradient, out = this.out.values, gradient = this.out.gradient;
            for(let i = 0; i < input.length; i += 1) {
                const derivative = gradient[i]; // dCost/dActivation
                input[i] = out[i] > 0 ? derivative : 0; // dCost/dInput = dActivation/dInput * dCost/dActivation
            }
        },
    };
    function RelU(properties) {
        const {  } = properties;
        const layer = addLayer(this, proto);
        layer.outShape = layer.inShape;
        layer.out = Vector(layer.outShape, 0);
        return layer;
    }
    return RelU;
})();

const LeakyRelU = (function() {
    const proto = {
        type: "leaky relu",
        forward() {
            const input = this.in.values, out = this.out.values;
            for(let i = 0; i < out.length; i += 1) {
                const v = input[i];
                out[i] = v > 0 ? v : v * this.slope;
            }
            return this.out;
        },
        backward() {
            const input = this.in.gradient, out = this.out.values, gradient = this.out.gradient;
            for(let i = 0; i < input.length; i += 1) {
                const derivative = gradient[i]; // dCost/dActivation
                input[i] = out[i] > 0 ? derivative : this.slope * derivative; // dCost/dInput = dActivation/dInput * dCost/dActivation
            }
        },
    };
    function LeakyRelU(properties) {
        const { slope = 0.01 } = properties;
        const layer = addLayer(this, proto);
        layer.outShape = layer.inShape;
        layer.out = Vector(layer.outShape, 0);
        layer.slope = slope;
        return layer;
    }
    return LeakyRelU;
})();

const Tanh = (function() {
    const proto = {
        type: "tanh",
        forward() {
            const input = this.in.values, out = this.out.values;
            for(let i = 0; i < out.length; i += 1) {
                out[i] = Math.tanh(input[i]);
            }
            return this.out;
        },
        backward() {
            const input = this.in.gradient, out = this.out.values, gradient = this.out.gradient;
            for(let i = 0; i < input.length; i += 1) {
                const derivative = gradient[i]; // dCost/dActivation
                const v = out[i];
                input[i] = (1 - v * v) * derivative; // dCost/dInput = dActivation/dInput * dCost/dActivation
            }
        },
    };
    function Tanh(properties) {
        const { } = properties;
        const layer = addLayer(this, proto);
        layer.outShape = layer.inShape;
        layer.out = Vector(layer.outShape, 0);
        return layer;
    }
    return Tanh;
})();

const Sigmoid = (function() {
    const proto = {
        type: "sigmoid",
        forward() {
            const input = this.in.values, out = this.out.values;
            for(let i = 0; i < out.length; i += 1) {
                out[i] = 1 / (1 + Math.exp(input[i]));
            }
            return this.out;
        },
        backward() {
            const input = this.in.gradient, out = this.out.values, gradient = this.out.gradient;
            for(let i = 0; i < input.length; i += 1) {
                const derivative = gradient[i]; // dCost/dActivation
                const v = out[i];
                input[i] = v * (v - 1) * derivative; // dCost/dInput = dActivation/dInput * dCost/dActivation
            }
        },
    };
    function Sigmoid(properties) {
        const { } = properties;
        const layer = addLayer(this, proto);
        layer.outShape = layer.inShape;
        layer.out = Vector(layer.outShape, 0);
        return layer;
    }
    return Sigmoid;
})();

const Softmax = (function() {
    const proto = {
        type: "softmax",
        forward() {
            const input = this.in.values, out = this.out.values;
            let a = 0;
            for(const v of input) { a += Math.exp(v); }
            this.a = a;
            for(let i = 0; i < out.length; i += 1) {
                out[i] = Math.exp(input[i]) / a;
            }
            return this.out;
        },
        backward() {
            const input = this.in.gradient, out = this.out.values, gradient = this.out.gradient;
            let a = this.a;
            for(let i = 0; i < out.length; i += 1) {
                const v = out[i]; // exp(input) / a
                input[i] = v * gradient[i];
                for(let j = 0; j < out.length; j += 1) {
                    input[i] -= v * out[j] * gradient[j]; // dCost/dInput = dActivation/dInput * dCost/dActivation
                }
            }
        },
    };
    function Softmax(properties) {
        const { } = properties;
        const layer = addLayer(this, proto);
        layer.outShape = layer.inShape;
        layer.out = Vector(layer.outShape, 0);
        return layer;
    }
    return Softmax;
})();

const GlobalAveragePooling = (function() {
    const proto = {
        type: "global average pooling",
        forward() {
            const out = this.out, input = this.in;
            const inputArea = input.values.length / input.shape[input.shape.length - 1], inverseInputArea = 1 / inputArea;
            let inputSliceIncrement = 1;
            for(let i = 0; i < input.shape.length - 1; i += 1) {
                inputSliceIncrement *= input.shape[i];
            }
            for(let i = 0; i < out.values.length; i += 1) {
                let average = 0;
                for(let j = i * inputSliceIncrement; j < inputArea + i * inputSliceIncrement; j += 1) {
                    average += input.values[j];
                }
                out.values[i] = average * inverseInputArea;
            }
            return out;
        },
        backward() {
            const out = this.out, input = this.in;
            const inputArea = input.values.length / input.shape[input.shape.length - 1], inverseInputArea = 1 / inputArea;
            let inputSliceIncrement = 1;
            for(let i = 0; i < input.shape.length - 1; i += 1) {
                inputSliceIncrement *= input.shape[i];
            }
            for(let i = 0; i < out.values.length; i += 1) {
                const derivative = out.gradient[i]; // dCost/dActivation
                for(let j = i * inputSliceIncrement; j < inputArea + i * inputSliceIncrement; j += 1) {
                    input.gradient[j] += inverseInputArea * derivative; // dCost/dInput = dActivation/dInput * dCost/dActivation
                }
            }
        },
    };
    function GlobalAveragePooling(properties) {
        const layer = addLayer(this, proto);
        const {  } = properties;
        layer.outShape = [layer.inShape[layer.inShape.length - 1]];
        layer.out = Vector(layer.outShape, 0);
        return layer;
    }
    return GlobalAveragePooling;
})();

const MaxPooling = (function() {
    const proto = {
        type: "max pooling",
        forward() {
            const dimensionality = this.kernelSize.length;
            const out = this.out;
            const filterPosition = [];
            let filterDepthMultiplier = 1;
            for(let i = 0; i < dimensionality; i += 1) {
                filterPosition[i] = -this.padding[i];
                filterDepthMultiplier *= this.kernelSize[i];
            }
            const outPosition = [];
            for(let i = 0; i < dimensionality + 1; i += 1) {
                outPosition[i] = 0;
            }
            const filterVolume = findVolume(this.kernelSize);
            let inDepthMultiplier = 1;
            const inputDimensionIncrements = [];
            for(let k = 0; k < dimensionality; k += 1) {
                inDepthMultiplier *= this.inShape[k];
                inputDimensionIncrements[k] = inDepthMultiplier;
            }

            // Feature maps
            for(let i = 0; i < out.values.length; i += 1) {
                let activation = -Infinity;
                const filterPixel = [];
                let inIndex = 0;
                for(let j = dimensionality - 1; j >= 0; j -= 1) {
                    filterPixel[j] = 0;
                    inIndex = filterPosition[j] + this.inShape[j] * inIndex;
                }
                let startIndex = inIndex;
                for(let j = 0; j < filterVolume; j += 1) {
                    // Input pixel with depth (filter #) and lth coordinate (filterPosition[l] + filterPixel[l] * dilation[l])
                    const index = inIndex + outPosition[dimensionality] * inDepthMultiplier;
                    const value = this.in.values[index] || 0;
                    if(value > activation) { activation = value; }

                    // Increment pixel of filter
                    filterPixel[0] += 1;
                    inIndex += this.dilation[0];
                    for(let k = 0; k < dimensionality - 1; k += 1) {
                        if(filterPixel[k] >= this.kernelSize[k]) {
                            for(let m = 0; m <= k; m += 1) {
                                filterPixel[m] = 0;
                            }
                            startIndex = inIndex = startIndex + inputDimensionIncrements[k];
                            filterPixel[k + 1] += 1;
                        }
                    }
                }
                out.values[i] = activation;

                // Move filter
                filterPosition[0] += this.stride[0];
                outPosition[0] += 1;
                for(let j = 0; j < dimensionality; j += 1) {
                    if(outPosition[j] >= this.outShape[j]) {
                        for(let k = 0; k <= j; k += 1) {
                            outPosition[k] = 0;
                            filterPosition[k] = -this.padding[k];
                        }
                        if(j < dimensionality - 1) { filterPosition[j + 1] += this.stride[j + 1]; }
                        outPosition[j + 1] += 1;
                    }
                }
            }
            return out;
        },
        backward() {
            const dimensionality = this.kernelSize.length;
            const out = this.out;
            const filterPosition = [];
            let filterDepthMultiplier = 1;
            for(let i = 0; i < dimensionality; i += 1) {
                filterPosition[i] = -this.padding[i];
                filterDepthMultiplier *= this.kernelSize[i];
            }
            const outPosition = [];
            for(let i = 0; i < dimensionality + 1; i += 1) {
                outPosition[i] = 0;
            }
            const filterVolume = findVolume(this.kernelSize);
            let inDepthMultiplier = 1;
            const inputDimensionIncrements = [];
            for(let k = 0; k < dimensionality; k += 1) {
                inDepthMultiplier *= this.inShape[k];
                inputDimensionIncrements[k] = inDepthMultiplier;
            }

            // Feature maps
            for(let i = 0; i < out.values.length; i += 1) {
                const derivative = this.out.gradient[i]; // dCost/dActivation
                const filterPixel = [];
                let inIndex = 0;
                for(let j = dimensionality - 1; j >= 0; j -= 1) {
                    filterPixel[j] = 0;
                    inIndex = filterPosition[j] + this.inShape[j] * inIndex;
                }
                let startIndex = inIndex;
                let maxIndex = 0, maxValue = -Infinity;
                for(let j = 0; j < filterVolume; j += 1) {
                    // Input pixel with depth (filter #) and lth coordinate (filterPosition[l] + filterPixel[l] * dilation[l])
                    const index = inIndex + outPosition[dimensionality] * inDepthMultiplier;
                    const value = this.in.values[index] || 0;
                    if(value > maxValue) {
                        maxIndex = index;
                        maxValue = value;
                    }

                    // Increment pixel of filter
                    filterPixel[0] += 1;
                    inIndex += this.dilation[0];
                    for(let k = 0; k < dimensionality - 1; k += 1) {
                        if(filterPixel[k] >= this.kernelSize[k]) {
                            for(let m = 0; m <= k; m += 1) {
                                filterPixel[m] = 0;
                            }
                            startIndex = inIndex = startIndex + inputDimensionIncrements[k];
                            filterPixel[k + 1] += 1;
                        }
                    }
                }
                this.in.gradient[maxIndex] += derivative; // dCost/dInput = dActivation/dInput * dCost/dActivation

                // Move filter
                filterPosition[0] += this.stride[0];
                outPosition[0] += 1;
                for(let j = 0; j < dimensionality; j += 1) {
                    if(outPosition[j] >= this.outShape[j]) {
                        for(let k = 0; k <= j; k += 1) {
                            outPosition[k] = 0;
                            filterPosition[k] = -this.padding[k];
                        }
                        if(j < dimensionality - 1) { filterPosition[j + 1] += this.stride[j + 1]; }
                        outPosition[j + 1] += 1;
                    }
                }
            }
        },
    };
    function MaxPooling(properties) {
        const layer = addLayer(this, proto);
        const dimensionality = layer.inShape.length - 1;
        const { kernelSize = Array(dimensionality).fill(1), stride = Array(dimensionality).fill(1), dilation = Array(dimensionality).fill(1), padding = Array(dimensionality).fill(0) } = properties;
        layer.kernelSize = kernelSize;
        layer.stride = stride;
        layer.dilation = dilation;
        layer.padding = padding;
        const out = [];
        for(let i = 0; i < dimensionality; i += 1) {
            const size = (layer.inShape[i] + padding[i] * 2 - kernelSize[i] * dilation[i]) / stride[i];
            if(size !== (size | 0)) { console.warn("MaxPooling input not divisible with current parameters; filters will not convolve across all values."); }
            out[i] = size + 1;
        }
        out[out.length] = layer.inShape[dimensionality];
        layer.outShape = out;
        layer.out = Vector(layer.outShape, 0);
        return layer;
    }
    return MaxPooling;
})();

const AveragePooling = (function() {
    const proto = {
        type: "average pooling",
        forward() {
            const dimensionality = this.kernelSize.length;
            const out = this.out;
            const filterPosition = [];
            let filterDepthMultiplier = 1;
            for(let i = 0; i < dimensionality; i += 1) {
                filterPosition[i] = -this.padding[i];
                filterDepthMultiplier *= this.kernelSize[i];
            }
            const outPosition = [];
            for(let i = 0; i < dimensionality + 1; i += 1) {
                outPosition[i] = 0;
            }
            const filterVolume = findVolume(this.kernelSize);
            let inDepthMultiplier = 1;
            const inputDimensionIncrements = [];
            for(let k = 0; k < dimensionality; k += 1) {
                inDepthMultiplier *= this.inShape[k];
                inputDimensionIncrements[k] = inDepthMultiplier;
            }

            // Feature maps
            for(let i = 0; i < out.values.length; i += 1) {
                let activation = 0;
                const filterPixel = [];
                let inIndex = 0;
                for(let j = dimensionality - 1; j >= 0; j -= 1) {
                    filterPixel[j] = 0;
                    inIndex = filterPosition[j] + this.inShape[j] * inIndex;
                }
                let startIndex = inIndex;
                for(let j = 0; j < filterVolume; j += 1) {
                    // Input pixel with depth (filter #) and lth coordinate (filterPosition[l] + filterPixel[l] * dilation[l])
                    const index = inIndex + outPosition[dimensionality] * inDepthMultiplier;
                    activation += (this.in.values[index] || 0) * this.invSum;

                    // Increment pixel of filter
                    filterPixel[0] += 1;
                    inIndex += this.dilation[0];
                    for(let k = 0; k < dimensionality - 1; k += 1) {
                        if(filterPixel[k] >= this.kernelSize[k]) {
                            for(let m = 0; m <= k; m += 1) {
                                filterPixel[m] = 0;
                            }
                            startIndex = inIndex = startIndex + inputDimensionIncrements[k];
                            filterPixel[k + 1] += 1;
                        }
                    }
                }
                out.values[i] = activation;

                // Move filter
                filterPosition[0] += this.stride[0];
                outPosition[0] += 1;
                for(let j = 0; j < dimensionality; j += 1) {
                    if(outPosition[j] >= this.outShape[j]) {
                        for(let k = 0; k <= j; k += 1) {
                            outPosition[k] = 0;
                            filterPosition[k] = -this.padding[k];
                        }
                        if(j < dimensionality - 1) { filterPosition[j + 1] += this.stride[j + 1]; }
                        outPosition[j + 1] += 1;
                    }
                }
            }
            return out;
        },
        backward() {
            const dimensionality = this.kernelSize.length;
            const out = this.out;
            const filterPosition = [];
            let filterDepthMultiplier = 1;
            for(let i = 0; i < dimensionality; i += 1) {
                filterPosition[i] = -this.padding[i];
                filterDepthMultiplier *= this.kernelSize[i];
            }
            const outPosition = [];
            for(let i = 0; i < dimensionality + 1; i += 1) {
                outPosition[i] = 0;
            }
            const filterVolume = findVolume(this.kernelSize);
            let inDepthMultiplier = 1;
            const inputDimensionIncrements = [];
            for(let k = 0; k < dimensionality; k += 1) {
                inDepthMultiplier *= this.inShape[k];
                inputDimensionIncrements[k] = inDepthMultiplier;
            }

            // Feature maps
            for(let i = 0; i < out.values.length; i += 1) {
                const derivative = this.out.gradient[i]; // dCost/dActivation
                const filterPixel = [];
                let inIndex = 0;
                for(let j = dimensionality - 1; j >= 0; j -= 1) {
                    filterPixel[j] = 0;
                    inIndex = filterPosition[j] + this.inShape[j] * inIndex;
                }
                let startIndex = inIndex;
                for(let j = 0; j < filterVolume; j += 1) {
                    // Input pixel with depth (filter #) and lth coordinate (filterPosition[l] + filterPixel[l] * dilation[l])
                    const index = inIndex + outPosition[dimensionality] * inDepthMultiplier;
                    this.in.gradient[index] += this.invSum * derivative; // dCost/dInput = dActivation/dInput * dCost/dActivation

                    // Increment pixel of filter
                    filterPixel[0] += 1;
                    inIndex += this.dilation[0];
                    for(let k = 0; k < dimensionality - 1; k += 1) {
                        if(filterPixel[k] >= this.kernelSize[k]) {
                            for(let m = 0; m <= k; m += 1) {
                                filterPixel[m] = 0;
                            }
                            startIndex = inIndex = startIndex + inputDimensionIncrements[k];
                            filterPixel[k + 1] += 1;
                        }
                    }
                }

                // Move filter
                filterPosition[0] += this.stride[0];
                outPosition[0] += 1;
                for(let j = 0; j < dimensionality; j += 1) {
                    if(outPosition[j] >= this.outShape[j]) {
                        for(let k = 0; k <= j; k += 1) {
                            outPosition[k] = 0;
                            filterPosition[k] = -this.padding[k];
                        }
                        if(j < dimensionality - 1) { filterPosition[j + 1] += this.stride[j + 1]; }
                        outPosition[j + 1] += 1;
                    }
                }
            }
        },
    };
    function AveragePooling(properties) {
        const layer = addLayer(this, proto);
        const dimensionality = layer.inShape.length - 1;
        const { kernelSize = Array(dimensionality).fill(1), stride = Array(dimensionality).fill(1), dilation = Array(dimensionality).fill(1), padding = Array(dimensionality).fill(0) } = properties;
        layer.kernelSize = kernelSize;
        layer.stride = stride;
        layer.dilation = dilation;
        layer.padding = padding;
        layer.invSum = 1 / findVolume(kernelSize);
        const out = [];
        for(let i = 0; i < dimensionality; i += 1) {
            const size = (layer.inShape[i] + padding[i] * 2 - kernelSize[i] * dilation[i]) / stride[i];
            if(size !== (size | 0)) { console.warn("AveragePooling input not divisible with current parameters; filters will not convolve across all values."); }
            out[i] = size + 1;
        }
        out[out.length] = layer.inShape[dimensionality];
        layer.outShape = out;
        layer.out = Vector(layer.outShape, 0);
        return layer;
    }
    return AveragePooling;
})();

function addLayer(obj, proto) {
    const layer = Object.create(proto);
    layer.inShape = obj.layers[obj.depth - 1].outShape;
    obj.depth += 1;
    obj.layers.push(layer);
    return layer;
}

const costFunctions = {
    "quadratic": (target, prediction) => 0.5 * (prediction - target) * (prediction - target),
};
const dCostFunctions = {
    "quadratic": (target, prediction) => (prediction - target),
}
const inputProto = {
    type: "input",
    forward() { return this.out = this.in; },
    backward() {},
};
const networkProto = {
    layers: [],
    depth: 1,

    forward(input, inputLayer = 0, outputLayer = this.depth - 1) {
        for(let i = inputLayer; i < outputLayer + 1; i += 1) {
            this.layers[i].in = input;
            input = this.layers[i].forward();
        }
        return input;
    },
    cost(target) {
        target = target.values;
        const layer = this.layers[this.depth - 1].out.values;
        let cost = 0;
        for(let i = 0; i < layer.length; i += 1) {
            cost += this.costFunction(target[i], layer[i]);
        }
        return cost;
    },
    backward(target, inputLayer = 0, outputLayer = this.depth - 1) {
        const out = this.layers[outputLayer].out;
        for(let i = 0; i < out.values.length; i += 1) {
            out.gradient[i] = this.dCostFunction(target.values[i], out.values[i]); // dCost/dActivation
        }
        for(let i = outputLayer; i >= inputLayer; i -= 1) {
            const parameters = this.layers[i].parameters;
            if(parameters) {
                for(const j of parameters) {
                    for(let k = 0; k < j.gradient.length; k += 1) {
                        j.gradient[k] = 0;
                    }
                }
            }
            for(let j = 0; j < this.layers[i].in.values.length; j += 1) {
                this.layers[i].in.gradient[j] = 0;
            }
            this.layers[i].backward();
        }
    },
    getGradients() {
        const gradients = [];
        for(const layer of this.layers) {
            const parameters = layer.parameters;
            if(!parameters) { continue; }
            for(const parameter of parameters) {
                for(const gradient of parameter.gradient) {
                    gradients.push(gradient);
                }
            }
        }
        return gradients;
    },
    sumGradients(array, gradients) {
        for(let i = 0; i < gradients.length; i += 1) {
            array[i] = (array[i] || 0) + (gradients[i] || 0);
        }
        return array;
    },
    descend(gradients, strength = 1) {
        let j = 0;
        for(const layer of this.layers) {
            const parameters = layer.parameters;
            if(!parameters) { continue; }
            for(const parameter of parameters) {
                for(let i = 0; i < parameter.values.length; i += 1) {
                    parameter.values[i] -= gradients[j] * strength;
                    j += 1;
                }
            }
        }
    },

    Dense, RelU, LeakyRelU, Tanh, Sigmoid, Softmax, Convolution, GlobalAveragePooling, MaxPooling, AveragePooling,
};
function Network(properties) {
    const { inputShape, costFunction, dCostFunction } = properties;
    const network = Object.create(networkProto);
    const input = Object.create(inputProto);
    input.inShape = inputShape;
    input.outShape = inputShape;
    network.layers.push(input);
    network.costFunction = (typeof costFunction === Function) ? costFunction : costFunctions[costFunction] || ((a, b) => (a - b) * (a - b));
    network.dCostFunction = (typeof costFunction === Function) ? dCostFunction : dCostFunctions[costFunction] || ((a, b) => 2 * (a - b));
    return network;
}

return { Network, Vector };
})();





