import { useId } from "react";
import { COMPONENTS, KIND_LABELS } from "./types";
import type { Assignment, Component, ConditionKind, Field, FieldErrors, Revision } from "./types";

export type BoundaryDraft = Revision & {
  name: string;
  kind: ConditionKind;
  components: Component[];
  expressions: Record<Component, string>;
};

export function boundaryDraft(assignment: Assignment, revision: Revision): BoundaryDraft {
  const components = assignment.components ?? [];
  const expressions = { x: "0", y: "0" };
  components.forEach((component, index) => {
    expressions[component] = assignment.values?.[index]?.expression ?? "0";
  });
  return {
    geometry_id: revision.geometry_id,
    draft_revision: revision.draft_revision,
    name: assignment.name,
    kind: assignment.kind,
    components,
    expressions,
  };
}

export function draftAssignment(region: string, draft: BoundaryDraft): Assignment {
  const assignment = { region, name: draft.name.trim(), kind: draft.kind };
  if (draft.kind === "traction_free") return assignment;
  if (draft.kind === "symmetry") return { ...assignment, components: draft.components };
  return {
    ...assignment,
    components: draft.components,
    values: draft.components.map((component) => ({ expression: draft.expressions[component].trim(), unit: "mm" })),
  };
}

type Props = {
  region: string;
  draft: BoundaryDraft;
  suggested: boolean;
  pending: boolean;
  errors?: FieldErrors;
  onEdit: (fields: Partial<BoundaryDraft>) => void;
};

export function BoundaryEditor({ region, draft, suggested, pending, errors = {}, onEdit }: Props) {
  const id = useId();
  const fieldId = (field: Field) => `${id}-${field}-error`;
  const fieldAttributes = (field: Field) => ({
    "aria-invalid": errors[field] ? true : undefined,
    "aria-describedby": errors[field] ? fieldId(field) : undefined,
  });
  const fieldError = (field: Field) => errors[field] && (
    <p id={fieldId(field)} className="boundary-field-error">{errors[field]}</p>
  );

  return (
    <section className="boundary-editor" data-testid="boundary-editor" data-region={region}>
      <h3>Boundary condition draft</h3>
      {suggested && <p>Suggested from geometry</p>}
      <div className="boundary-field">
        <label htmlFor={`${id}-name`}>Condition Name</label>
        <input id={`${id}-name`} value={draft.name} {...fieldAttributes("name")}
          onChange={(event) => onEdit({ name: event.target.value })} />
        {fieldError("name")}
      </div>
      <div className="boundary-field">
        <label htmlFor={`${id}-kind`}>Boundary Condition Type</label>
        <select id={`${id}-kind`} value={draft.kind} {...fieldAttributes("kind")}
          onChange={(event) => onEdit({ kind: event.target.value as ConditionKind })}>
          {Object.entries(KIND_LABELS).map(([kind, label]) => <option key={kind} value={kind}>{label}</option>)}
        </select>
        {fieldError("kind")}
      </div>
      {draft.kind !== "traction_free" && (
        <fieldset {...fieldAttributes("components")}>
          <legend>Displacement components</legend>
          {COMPONENTS.map((component) => {
            const checked = draft.components.includes(component);
            return (
              <div className="boundary-field" key={component}>
                <label className="boundary-component">
                  <input type="checkbox" checked={checked} onChange={(event) => onEdit({
                    components: event.target.checked
                      ? [...draft.components, component]
                      : draft.components.filter((item) => item !== component),
                  })} />
                  Constrain {component}
                </label>
                {draft.kind === "dirichlet" && checked && <>
                  <label htmlFor={`${id}-${component}`}>u_{component} expression (mm)</label>
                  <input id={`${id}-${component}`} value={draft.expressions[component]} {...fieldAttributes(component)}
                    onChange={(event) => onEdit({ expressions: { ...draft.expressions, [component]: event.target.value } })} />
                  {fieldError(component)}
                </>}
              </div>
            );
          })}
        </fieldset>
      )}
      {fieldError("components")}
      {pending && <p>Unapplied draft</p>}
    </section>
  );
}
