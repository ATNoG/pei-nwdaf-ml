from fastapi import APIRouter, HTTPException, Request
import logging

from src.schemas.data import DataQueryRequest, DataStorageRequest, LatencyQueryRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query")
async def query_training_data(req: DataQueryRequest, request: Request):
    """
    Query training data from Data Storage component

    This endpoint fetches data that will be used for model training.
    """
    ml_interface = request.app.state.ml_interface

    if not ml_interface:
        raise HTTPException(status_code=500, detail="ML Interface not initialized")

    try:
        logger.info(f"Querying training data: {req.data_type} from {req.start_time} to {req.end_time}")

        data = await ml_interface.get_training_data_async(
            start_time=req.start_time,
            end_time=req.end_time,
            data_type=req.data_type,
            filters=req.filters
        )

        if data is None:
            raise HTTPException(
                status_code=503,
                detail="Failed to retrieve data from Data Storage API"
            )

        return {
            "status": "success",
            "data": data,
            "query": {
                "start_time": req.start_time,
                "end_time": req.end_time,
                "data_type": req.data_type,
                "filters": req.filters
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying training data: {e}")
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


@router.post("/query/latency")
async def query_latency_data(req: LatencyQueryRequest, request: Request):
    """
    Query latency data from Data Storage component

    This endpoint fetches processed latency data from the Data Storage API's
    /api/v1/processed/latency/ endpoint for model training or analysis.

    Required parameters:
    - start_time: Window start time in ISO format
    - end_time: Window end time in ISO format
    - cell_index: Cell index to query

    Optional parameters:
    - offset: Number of records to skip (default: 0)
    - limit: Maximum number of records to return (default: 100, max: 1000)
    """
    ml_interface = request.app.state.ml_interface

    if not ml_interface:
        raise HTTPException(status_code=500, detail="ML Interface not initialized")

    try:
        logger.info(f"Querying latency data for cell {req.cell_index} from {req.start_time} to {req.end_time}")

        # Build query parameters matching the Data Storage API requirements
        params = {
            "start_time": req.start_time,
            "end_time": req.end_time,
            "cell_index": req.cell_index,
            "offset": req.offset,
            "limit": req.limit
        }

        # Query the Data Storage API /api/v1/processed/latency/ endpoint
        data = await ml_interface.request_data_from_storage_async(
            endpoint="/api/v1/processed/latency/",
            params=params,
            method="GET"
        )

        if data is None:
            raise HTTPException(
                status_code=503,
                detail="Failed to retrieve latency data from Data Storage API"
            )

        return {
            "status": "success",
            "data": data,
            "query": {
                "start_time": req.start_time,
                "end_time": req.end_time,
                "cell_index": req.cell_index,
                "offset": req.offset,
                "limit": req.limit
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying latency data: {e}")
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


@router.post("/request")
async def request_from_storage(req: DataStorageRequest, request: Request):
    """
    Generic endpoint to request data from Data Storage API

    Allows flexible interaction with any Data Storage endpoint.
    """
    ml_interface = request.app.state.ml_interface

    if not ml_interface:
        raise HTTPException(status_code=500, detail="ML Interface not initialized")

    try:
        logger.info(f"Custom data storage request: {req.method} {req.endpoint}")

        result = await ml_interface.request_data_from_storage_async(
            endpoint=req.endpoint,
            params=req.params,
            method=req.method,
            data=req.data,
            timeout=req.timeout
        )

        if result is None:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to request data from endpoint: {req.endpoint}"
            )

        return {
            "status": "success",
            "result": result,
            "request": {
                "endpoint": req.endpoint,
                "method": req.method
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error requesting from storage: {e}")
        raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")


@router.get("/storage/status")
async def get_storage_status(request: Request):
    """
    Check the status of Data Storage API connection
    """
    ml_interface = request.app.state.ml_interface

    if not ml_interface:
        raise HTTPException(status_code=500, detail="ML Interface not initialized")

    try:
        connected = ml_interface.check_data_storage_connection()
        component_status = ml_interface.get_component_status('data_storage')

        return {
            "status": "success",
            "connected": connected,
            "api_url": ml_interface.data_storage_api_url,
            "component_status": component_status
        }

    except Exception as e:
        logger.error(f"Error checking storage status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/storage/test")
async def test_storage_connection(request: Request):
    """
    Test the connection to Data Storage API
    """
    ml_interface = request.app.state.ml_interface

    if not ml_interface:
        raise HTTPException(status_code=500, detail="ML Interface not initialized")

    try:
        result = await ml_interface.request_data_from_storage_async(
            endpoint="/health",
            timeout=5
        )

        if result is None:
            result = await ml_interface.request_data_from_storage_async(
                endpoint="/",
                timeout=5
            )

        if result is not None:
            return {
                "status": "success",
                "message": "Successfully connected to Data Storage API",
                "api_url": ml_interface.data_storage_api_url,
                "response": result
            }
        else:
            raise HTTPException(
                status_code=503,
                detail="Data Storage API not reachable"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing storage connection: {e}")
        raise HTTPException(status_code=500, detail=str(e))
