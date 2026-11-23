import type { FrontendRenderer, FrontendRendererArgs } from "@streamlit/component-v2-lib";
import { createRoot, type Root } from "react-dom/client";
import { BoundaryPicker } from "./BoundaryPicker";
import type { ApplyEvent, PickerData } from "./types";
import "./styles.css";

const roots = new WeakMap<FrontendRendererArgs["parentElement"], Root>();

const mount: FrontendRenderer<{ apply: ApplyEvent }, PickerData> = ({
  data, parentElement, setTriggerValue,
}) => {
  const element = parentElement.querySelector(".react-root");
  if (!element) throw new Error("Boundary Picker mount element is missing.");

  let root = roots.get(parentElement);
  if (!root) {
    root = createRoot(element);
    roots.set(parentElement, root);
  }
  root.render(<BoundaryPicker data={data} onApply={(event) => setTriggerValue("apply", event)} />);

  return () => {
    roots.get(parentElement)?.unmount();
    roots.delete(parentElement);
  };
};

export default mount;
