def prepare_goatbench_navigation_goals(
    scene_name,
    episode,
    all_navigation_goals,
):
    # filter the task
    # code borrowed from goatbench
    filtered_tasks = []
    for goal in episode["tasks"]:
        goal_type = goal[1]
        goal_category = goal[0]
        goal_inst_id = goal[2]

        dset_same_cat_goals = [
            x
            for x in all_navigation_goals.values()
            if x[0]["object_category"] == goal_category
        ]

        if goal_type == "description":
            goal_inst = [
                x for x in dset_same_cat_goals[0] if x["object_id"] == goal_inst_id
            ]
            if len(goal_inst[0]["lang_desc"].split(" ")) <= 55:
                filtered_tasks.append(goal)
        else:
            filtered_tasks.append(goal)

    all_subtask_goals = []
    all_subtask_goal_types = []
    for goal in filtered_tasks:
        goal_type = goal[1]
        goal_category = goal[0]
        goal_inst_id = goal[2]

        all_subtask_goal_types.append(goal_type)

        dset_same_cat_goals = [
            x
            for x in all_navigation_goals.values()
            if x[0]["object_category"] == goal_category
        ]
        children_categories = dset_same_cat_goals[0][0]["children_object_categories"]
        for child_category in children_categories:
            goal_key = f"{scene_name}.basis.glb_{child_category}"
            if goal_key not in all_navigation_goals:
                print(f"!!! {goal_key} not in navigation_goals")
                continue
            print(f"!!! {goal_key} added")
            dset_same_cat_goals[0].extend(all_navigation_goals[goal_key])

        assert (
            len(dset_same_cat_goals) == 1
        ), f"more than 1 goal categories for {goal_category}"

        if goal_type == "object":
            all_subtask_goals.append(dset_same_cat_goals[0])
        else:
            goal_inst = [
                x for x in dset_same_cat_goals[0] if x["object_id"] == goal_inst_id
            ]
            all_subtask_goals.append(goal_inst)

    return all_subtask_goal_types, all_subtask_goals
