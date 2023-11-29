"""Functions to launch, retrieve, and parse specific EasyTurk tasks.
"""
import json

from easyturk import EasyTurk

def launch_lyft_command_anno(data, reward=1.00, tasks_per_hit=10, sandbox=False,
                              max_assignments=1, hits_approved=10000, percent_approved=95, title='Self-Driving Car - Path to follow',
                                description='Draw the path that the car should follow.',
                              duration=1200, use_masters=False):
    """Launches HITs to ask workers to caption objects in images.

    Args:
        data: List containing image urls for the task.
        reward: A postive valued dollar amount per task.
        tasks_per_hit: Number of images per hit.
        sandbox: Whether to interact on sandbox or production.

    Returns:
        A list of hit ids that have been launched.
    """
    et = EasyTurk(sandbox=sandbox)
    template = 'gt_lyft_path_annotation.html'
    hit_ids = []
    i = 0
    while i < len(data):
        hit = et.launch_hit(
            template, data[i], reward=reward,
            title=title,
            description=description,
            keywords='image, path, self-driving car',hits_approved=hits_approved,percent_approved=95,
            max_assignments=max_assignments, duration=duration, use_masters=use_masters)
        hit_id = hit['HIT']['HITId']
        hit_ids.append(hit_id)
        i += tasks_per_hit
    return hit_ids


def launch_t2c_intent_only_qualified(data, qualification, reward=1.00, tasks_per_hit=10, sandbox=False,
                              max_assignments=1, title='Self-Driving Car - Path to follow',
                              duration=1200):
    """Launches HITs to ask workers to caption objects in images.

    Args:
        data: List containing image urls for the task.
        reward: A postive valued dollar amount per task.
        tasks_per_hit: Number of images per hit.
        sandbox: Whether to interact on sandbox or production.

    Returns:
        A list of hit ids that have been launched.
    """
    et = EasyTurk(sandbox=sandbox)
    template = 'annotate_command_intent.html'
    hit_ids = []
    i = 0
    try:
        while i < len(data):
            hit = et.launch_hit(
                template, data[i:i+tasks_per_hit], reward=reward,
                title=title,
                description='Indicate the intent of the command',
                keywords="Self-Driving Car, Path annotations",
                max_assignments=max_assignments, duration=duration,
                qualification=qualification)
            hit_id = hit['HIT']['HITId']
            hit_ids.append(hit_id)
            i += tasks_per_hit
    except Exception as e:
        print("crashing! Already launched", hit_ids)
        json.dump(hit_ids, open("launched_intent_hits.json", "w"))
        json.dump({"error": str(e)}, open("error_intent_message.json", "w"))

    return hit_ids


def launch_t2c_path_anno_only_qualified(data, qualification, reward=1.00, tasks_per_hit=10, sandbox=False,
                              max_assignments=1, title='Self-Driving Car - Path to follow',
                              duration=1200):
    """Launches HITs to ask workers to caption objects in images.

    Args:
        data: List containing image urls for the task.
        reward: A postive valued dollar amount per task.
        tasks_per_hit: Number of images per hit.
        sandbox: Whether to interact on sandbox or production.

    Returns:
        A list of hit ids that have been launched.
    """
    et = EasyTurk(sandbox=sandbox)
    template = 'annotate_path.html'
    hit_ids = []
    i = 0
    try:
        while i < len(data):
            hit = et.launch_hit(
                template, data[i:i+tasks_per_hit], reward=reward,
                title=title,
                description='Draw path for the car',
                keywords="Self-Driving Car, Path annotations",
                max_assignments=max_assignments, duration=duration,
                qualification=qualification)
            hit_id = hit['HIT']['HITId']
            hit_ids.append(hit_id)
            i += tasks_per_hit
    except Exception as e:
        print("crashing! Already launched", hit_ids)
        json.dump(hit_ids, open("launched_hits.json", "w"))
        json.dump({"error": str(e)}, open("error_message.json", "w"))

    return hit_ids


def launch_lyft_anno_only_qualified(data, qualification, reward=1.00,  sandbox=False,
                              max_assignments=1, hits_approved=10000, percent_approved=95, title='Self-Driving Car - Path to follow',
                                description='Give commands to a self-driving car.',
                              duration=1200, use_masters=False):
    """Launches HITs to ask workers to caption objects in images.

    Args:
        data: List containing image urls for the task.
        reward: A postive valued dollar amount per task.
        tasks_per_hit: Number of images per hit.
        sandbox: Whether to interact on sandbox or production.

    Returns:
        A list of hit ids that have been launched.
    """
    et = EasyTurk(sandbox=sandbox)
    template = 'gt_lyft_path_annotation.html'
    hit_ids = []
    try:
        for d in data:
            hit = et.launch_hit(
                template, d, reward=reward,
                title=title,
                description=description,
                keywords='image, path, self-driving car', hits_approved=hits_approved, percent_approved=percent_approved,
                max_assignments=max_assignments, duration=duration, use_masters=use_masters, qualification=qualification)
            hit_id = hit['HIT']['HITId']
            hit_ids.append(hit_id)
        #
        # hit = et.launch_hit(
        #     template, data, reward=reward,
        #     title=title,
        #     description='Give commands for the drawn path',
        #     keywords='image, path, self-driving car', max_assignments=max_assignments, duration=duration,
        #     qualification=qualification)
        #
        # hit_id = hit['HIT']['HITId']
        # hit_ids.append(hit_id)
        # i += tasks_per_hit
    except Exception as e:
        print("error", e)
        print("crashing! Already launched", hit_ids)
        json.dump(hit_ids, open("launched_lyft_hits.json", "w"))
        json.dump({"error": str(e)}, open("error_lyft_message.json", "w"))

    return hit_ids


def launch_t2c_path_anno(data, reward=1.00, tasks_per_hit=10, sandbox=False,
                              max_assignments=1, hits_approved=10000, percent_approved=95, title='Self-Driving Car - Path to follow',
                                description='Draw the path that the car should follow.',
                              duration=1200, use_masters=False):
    """Launches HITs to ask workers to caption objects in images.

    Args:
        data: List containing image urls for the task.
        reward: A postive valued dollar amount per task.
        tasks_per_hit: Number of images per hit.
        sandbox: Whether to interact on sandbox or production.

    Returns:
        A list of hit ids that have been launched.
    """
    et = EasyTurk(sandbox=sandbox)
    template = 'annotate_path.html'
    hit_ids = []
    i = 0
    while i < len(data):
        hit = et.launch_hit(
            template, data[i:i+tasks_per_hit], reward=reward,
            title=title,
            description=description,
            keywords='image, path, self-driving car',hits_approved=hits_approved,percent_approved=95,
            max_assignments=max_assignments, duration=duration, use_masters=use_masters)
        hit_id = hit['HIT']['HITId']
        hit_ids.append(hit_id)
        i += tasks_per_hit
    return hit_ids

def launch_verify_question_answer(data, reward=1.00, tasks_per_hit=50, sandbox=False):
    """Launches HITs to ask workers to verify bounding boxes.

    Args:
        data: List containing image urls, questions and answers, for the task.
        reward: A postive valued dollar amount per task.
        tasks_per_hit: Number of images per hit.
        sandbox: Whether to interact on sandbox or production.

    Returns:
        A list of hit ids that have been launched.
    """
    et = EasyTurk(sandbox=sandbox)
    template = 'verify_question_answer.html'
    hit_ids = []
    i = 0
    while i < len(data):
        hit = et.launch_hit(
            template, data[i:i+tasks_per_hit], reward=reward,
            title='Verify the answer to a question about an picture',
            description=('Verify whether an answer to a question about a picture is correct.'),
            keywords='image, text, picture, answer, question, relationship')
        hit_id = hit['HIT']['HITId']
        hit_ids.append(hit_id)
        i += tasks_per_hit
    return hit_ids

def launch_verify_relationship(data, reward=1.00, tasks_per_hit=30, sandbox=False):
    """Launches HITs to ask workers to verify bounding boxes.

    Args:
        data: List containing image urls, relationships, for the task.
        reward: A postive valued dollar amount per task.
        tasks_per_hit: Number of images per hit.
        sandbox: Whether to interact on sandbox or production.

    Returns:
        A list of hit ids that have been launched.
    """
    et = EasyTurk(sandbox=sandbox)
    template = 'verify_relationship.html'
    hit_ids = []
    i = 0
    while i < len(data):
        hit = et.launch_hit(
            template, data[i:i+tasks_per_hit], reward=reward,
            title='Verify relationships between objects in pictures',
            description=('Verify whether the relationships are correctly identified in pictures.'),
            keywords='image, text, picture, object, bounding box, relationship')
        hit_id = hit['HIT']['HITId']
        hit_ids.append(hit_id)
        i += tasks_per_hit
    return hit_ids


def launch_verify_bbox(data, reward=1.00, tasks_per_hit=30, sandbox=False):
    """Launches HITs to ask workers to verify bounding boxes.

    Args:
        data: List containing image urls, objects, for the task.
        reward: A postive valued dollar amount per task.
        tasks_per_hit: Number of images per hit.
        sandbox: Whether to interact on sandbox or production.

    Returns:
        A list of hit ids that have been launched.
    """
    et = EasyTurk(sandbox=sandbox)
    template = 'verify_bbox.html'
    hit_ids = []
    i = 0
    while i < len(data):
        hit = et.launch_hit(
            template, data[i:i+tasks_per_hit], reward=reward,
            title='Verify objects in pictures',
            description=('Verify whether objects are correctly identified in pictures.'),
            keywords='image, text, picture, object, bounding box')
        hit_id = hit['HIT']['HITId']
        hit_ids.append(hit_id)
        i += tasks_per_hit
    return hit_ids


def launch_caption(data, reward=1.00, tasks_per_hit=10, sandbox=False):
    """Launches HITs to ask workers to caption images.

    Args:
        data: List containing image urls for the task.
        reward: A postive valued dollar amount per task.
        tasks_per_hit: Number of images per hit.
        sandbox: Whether to interact on sandbox or production.

    Returns:
        A list of hit ids that have been launched.
    """
    et = EasyTurk(sandbox=sandbox)
    template = 'write_caption.html'
    hit_ids = []
    i = 0
    while i < len(data):
        hit = et.launch_hit(
            template, data[i:i+tasks_per_hit], reward=reward,
            title='Caption some pictures',
            description=('Write captions about the contents of images.'),
            keywords='image, caption, text')
        hit_id = hit['HIT']['HITId']
        hit_ids.append(hit_id)
        i += tasks_per_hit
    return hit_ids


def fetch_completed_hits(hit_ids, approve=True, sandbox=False):
    """Grabs the results for the hit ids.

    Args:
        hit_ids: A list of hit ids to fetch.
        approve: Whether to approve the hits that have been submitted.
        sandbox: Whether to interact on sandbox or production.

    Returns:
        A dictionary from hit_id to the result, if that hit_id has
        been submitted.
    """
    et = EasyTurk(sandbox=sandbox)
    output = {}
    for hit_id in hit_ids:
        results = et.get_results(hit_id, reject_on_fail=False)
        if len(results) > 0:
            output[hit_id] = results
            if approve:
                for assignment in results:
                    assignment_id = assignment['assignment_id']
                    et.approve_assignment(assignment_id)
    return output
