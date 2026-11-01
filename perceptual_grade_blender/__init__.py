"""Perceptual Grade — perceptual color grading with interactive scopes."""
RELEASE = '1.23.2'


def register():
    from .integration.lifecycle import registered_session
    previous = registered_session()
    if previous is not None:
        previous.close()
        if not previous.closed or not previous.app.closed:
            raise RuntimeError('Previous Perceptual Grade lifecycle has not reached terminal cleanup')
    from .machines import create_machines
    from .session import Session
    Session(create_machines()).register()


def unregister():
    from .integration.lifecycle import registered_session
    session = registered_session()
    if session is not None:
        session.close()
